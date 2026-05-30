"""Anomaly Detector: VAE + Isolation Forest hybrid for zero-day detection.

VAE learns normal traffic baseline; Isolation Forest provides complementary
anomaly signal. Hybrid score = vae_weight * vae_score + iforest_weight * iforest_score.
Trained ONLY on normal traffic (cursorrules).
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from tensorflow import keras

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger
from src.utils.seed import get_seed, set_all_seeds

logger = get_logger(__name__)


def _build_vae(
    input_dim: int,
    latent_dim: int,
    encoder_units: list[int],
    decoder_units: list[int],
) -> tuple[keras.Model, keras.Model]:
    """Build VAE encoder and decoder."""
    encoder_input = keras.Input(shape=(input_dim,), name="encoder_input")
    x = encoder_input
    for i, u in enumerate(encoder_units):
        x = keras.layers.Dense(u, activation="relu", name=f"enc_dense_{i}")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)

    mu = keras.layers.Dense(latent_dim, name="mu")(x)
    log_var = keras.layers.Dense(latent_dim, name="log_var")(x)

    def sampling(args: tuple) -> tf.Tensor:
        mu_t, log_var_t = args
        log_var_t = tf.clip_by_value(log_var_t, -10.0, 10.0)
        eps = tf.random.normal(tf.shape(mu_t))
        return mu_t + tf.exp(0.5 * log_var_t) * eps

    z = keras.layers.Lambda(sampling, name="z")([mu, log_var])
    encoder = keras.Model(encoder_input, [mu, log_var, z], name="encoder")

    decoder_input = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = decoder_input
    for i, u in enumerate(decoder_units):
        x = keras.layers.Dense(u, activation="relu", name=f"dec_dense_{i}")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)
    decoder_output = keras.layers.Dense(input_dim, activation="linear", name="decoder_output")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    return encoder, decoder


class VAEModel(keras.Model):
    """VAE with custom train_step for reconstruction + KL loss."""

    def __init__(self, encoder: keras.Model, decoder: keras.Model, kl_weight: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data: tuple) -> dict:
        x, y = data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x)
            log_var = tf.clip_by_value(log_var, -10.0, 10.0)
            recon = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.square(x - recon))
            kl_loss = -0.5 * tf.reduce_mean(
                1.0 + log_var - tf.square(mu) - tf.exp(log_var)
            )
            total_loss = recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def test_step(self, data: tuple) -> dict:
        x, y = data
        mu, log_var, z = self.encoder(x)
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        recon = self.decoder(z)
        recon_loss = tf.reduce_mean(tf.square(x - recon))
        kl_loss = -0.5 * tf.reduce_mean(
            1.0 + log_var - tf.square(mu) - tf.exp(log_var)
        )
        total_loss = recon_loss + self.kl_weight * kl_loss
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}


class AnomalyDetector:
    """VAE + Isolation Forest hybrid anomaly detector.

    Fit on normal traffic only. Predict returns hybrid anomaly score [0, 1].
    """

    def __init__(self) -> None:
        self._config = load_config("anomaly_detector")
        self._vae: keras.Model | None = None
        self._encoder: keras.Model | None = None
        self._decoder: keras.Model | None = None
        self._iforest: IsolationForest | None = None
        self._vae_threshold: float = 0.0
        self._input_dim: int = 0
        self._fitted = False

    def build(self, input_dim: int) -> "AnomalyDetector":
        """Build VAE model without training. Optional; fit() calls this internally.

        Args:
            input_dim: Number of input features.

        Returns:
            self for chaining.
        """
        self._input_dim = input_dim
        self._vae = self._build_vae_model(input_dim)
        logger.info("VAE built: input_dim=%s", input_dim)
        return self

    def _get_kl_weight(self) -> float:
        """Read kl_weight from config. Never hardcode."""
        return float(self._config.get("vae", {}).get("kl_weight", 0.001))

    def _build_vae_model(self, input_dim: int) -> keras.Model:
        """Build and compile full VAE with custom train_step."""
        vae_cfg = self._config.get("vae", {})
        encoder, decoder = _build_vae(
            input_dim=input_dim,
            latent_dim=vae_cfg.get("latent_dim", 32),
            encoder_units=vae_cfg.get("encoder_units", [256, 128, 64]),
            decoder_units=vae_cfg.get("decoder_units", [64, 128, 256]),
        )
        self._encoder = encoder
        self._decoder = decoder

        vae = VAEModel(
            encoder=encoder,
            decoder=decoder,
            kl_weight=self._get_kl_weight(),
            name="vae",
        )
        vae.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=vae_cfg.get("learning_rate", 0.001),
                clipnorm=1.0,
            ),
        )
        return vae

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        normal_label: int | str = 0,
    ) -> "AnomalyDetector":
        """Fit VAE and Isolation Forest on NORMAL traffic only.

        Args:
            X: Features (n_samples, n_features).
            y: Labels. If provided, only normal_label samples are used.
            normal_label: Class index or name for "normal" traffic.

        Returns:
            self for chaining.
        """
        set_all_seeds()
        cfg = self._config
        if not cfg.get("training", {}).get("train_on_normal_only", True):
            logger.warning("train_on_normal_only is False - anomaly detector should use normal only")

        if y is not None:
            mask = np.array(y) == normal_label
            X = X[mask]
            logger.info("Filtered to normal traffic: %s samples", len(X))

        if len(X) == 0:
            raise ValueError("No normal samples to fit. Check y and normal_label.")

        self._input_dim = X.shape[1]
        vae_cfg = cfg.get("vae", {})
        iforest_cfg = cfg.get("isolation_forest", {})
        ensemble_cfg = cfg.get("ensemble", {})

        self._vae = self._build_vae_model(self._input_dim)
        self._vae.fit(
            X,
            X,
            epochs=vae_cfg.get("epochs", 30),
            batch_size=vae_cfg.get("batch_size", 512),
            validation_split=0.1,
            verbose=1,
        )

        recon = self._vae.predict(X, verbose=0)
        recon_errors = np.mean(np.square(X - recon), axis=1)
        self._vae_threshold = np.percentile(
            recon_errors,
            vae_cfg.get("threshold_percentile", 95),
        )
        logger.info("VAE threshold (%%95 recon error): %.6f", self._vae_threshold)

        self._iforest = IsolationForest(
            n_estimators=iforest_cfg.get("n_estimators", 100),
            contamination=iforest_cfg.get("contamination", 0.05),
            random_state=iforest_cfg.get("random_state", get_seed()),
            max_features=iforest_cfg.get("max_features", 1.0),
        )
        self._iforest.fit(X)
        logger.info("Isolation Forest fitted on %s samples", len(X))

        self._fitted = True
        return self

    def _vae_score(self, X: np.ndarray) -> np.ndarray:
        """VAE anomaly score [0, 1]. Higher = more anomalous."""
        recon = self._vae.predict(X, verbose=0)
        recon_errors = np.mean(np.square(X - recon), axis=1)
        scores = np.minimum(1.0, recon_errors / (self._vae_threshold + 1e-8))
        return scores.astype(np.float32)

    def _iforest_score(self, X: np.ndarray) -> np.ndarray:
        """Isolation Forest anomaly score [0, 1]. Higher = more anomalous."""
        dec = self._iforest.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(-dec))
        return scores.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hybrid anomaly score [0, 1] per sample.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Anomaly scores. Values above final_threshold → anomaly.
        """
        if not self._fitted:
            raise RuntimeError("AnomalyDetector not fitted. Call fit() first.")

        ensemble_cfg = self._config.get("ensemble", {})
        vae_w = ensemble_cfg.get("vae_weight", 0.6)
        if_w = ensemble_cfg.get("iforest_weight", 0.4)

        vae_s = self._vae_score(X)
        if_s = self._iforest_score(X)
        return (vae_w * vae_s + if_w * if_s).astype(np.float32)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Binary labels: 0=normal, 1=anomaly."""
        scores = self.predict(X)
        threshold = self._config.get("ensemble", {}).get("final_threshold", 0.5)
        return (scores >= threshold).astype(np.int32)

    def save(self, path: Path | None = None) -> None:
        """Save VAE, IForest, and metadata."""
        if not self._fitted:
            raise RuntimeError("Nothing to save. Call fit() first.")

        save_path = path or get_project_root() / "artifacts" / "models" / "anomaly_detector"
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        enc_in = self._encoder.input
        z = self._encoder(enc_in)[2]
        inference_model = keras.Model(inputs=enc_in, outputs=self._decoder(z))
        inference_model.save(save_path / "vae.keras")
        joblib.dump(
            {
                "iforest": self._iforest,
                "vae_threshold": self._vae_threshold,
                "input_dim": self._input_dim,
            },
            save_path / "metadata.pkl",
        )
        logger.info("AnomalyDetector saved to %s", save_path)

    def load(self, path: Path | None = None) -> "AnomalyDetector":
        """Load VAE, IForest, and metadata."""
        load_path = path or get_project_root() / "artifacts" / "models" / "anomaly_detector"
        load_path = Path(load_path)

        self._vae = keras.models.load_model(load_path / "vae.keras")
        data = joblib.load(load_path / "metadata.pkl")
        self._iforest = data["iforest"]
        self._vae_threshold = data["vae_threshold"]
        self._input_dim = data["input_dim"]
        self._fitted = True
        logger.info("AnomalyDetector loaded from %s", load_path)
        return self
