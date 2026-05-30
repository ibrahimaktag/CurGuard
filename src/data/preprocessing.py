import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from abc import ABC, abstractmethod
from sklearn.preprocessing import RobustScaler, LabelEncoder

def setup_logger(name: str) -> logging.Logger:
    """Returns a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class BasePreprocessor(ABC):
    """
    Base class containing common preprocessing logic for all network traffic datasets.
    Implements fit_transform (training) and transform (inference) separation to prevent data leakage.
    """
    
    NON_FEATURE_COLS = [
        'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'Timestamp', 
        'Flow ID', 'Unnamed: 0', 'id', 'ID', 'index', 'source_file'
    ]

    def __init__(self, target_col: str, task_type: str = 'binary', artifact_dir: str = '.'):
        """
        Args:
            target_col (str): The name of the target column.
            task_type (str): The classification task type ('binary' or 'multiclass').
            artifact_dir (str): Directory where Joblib artifact files will be saved.
        """
        self.target_col = target_col
        self.task_type = task_type
        self.artifact_dir = Path(artifact_dir)
        self.logger = setup_logger(self.__class__.__name__)
        
        # State Variables
        self.medians_: Dict[str, float] = {}
        self.scaler_: Optional[RobustScaler] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.dropped_corr_cols_: List[str] = []
        self.numeric_features_: List[str] = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies common data cleaning steps."""
        self.logger.info("Temel veri temizleme işlemleri (Duplicates, Infs, Sızıntı sütunları) uygulanıyor...")
        
        # 1. Drop metadata columns that cause data leakage
        cols_to_drop = [c for c in self.NON_FEATURE_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.logger.info(f"Kaldırılan meta veri sütunları: {cols_to_drop}")

        # 2. Convert infinities to NaN globally
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 3. Drop duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            self.logger.info(f"{dropped_rows} adet mükerrer (duplicate) satır silindi.")
            
        return df

    def reduce_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Expert Recommendation] Optimizes RAM usage for massive datasets (Downcasting).
        Reduces data types without precision loss.
        """
        self.logger.info("Bellek optimizasyonu (Downcasting) başlatılıyor...")
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if pd.api.types.is_integer_dtype(col_type):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        # float16 can sometimes cause instability in machine learning models; float32 is generally safer.
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                        
        end_mem = df.memory_usage().sum() / 1024**2
        self.logger.info(f"Bellek kullanımı {start_mem:.2f} MB'den {end_mem:.2f} MB'ye düştü (Azalma: {100 * (start_mem - end_mem) / start_mem:.1f}%).")
        return df

    def _coerce_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force every non-target column to a numeric dtype.

        This guard step runs immediately after ``clean_data()`` so that any
        residual object-type values (mixed strings from malformed CSV rows,
        column headers accidentally read as data, etc.) are converted before
        the rest of the pipeline touches them.

        Conversion strategy
        -------------------
        * ``pd.to_numeric(errors='coerce')`` silently turns un-parseable
          strings into NaN instead of raising.  These NaN cells are then
          caught and filled by the subsequent ``handle_missing()`` step using
          per-column medians computed from the training set.
        * Columns that are *already* numeric are cast to ``float32`` to
          guarantee PyTorch tensor compatibility regardless of how
          ``reduce_memory_usage()`` downcast them.
        * The target column is always excluded to avoid interfering with
          ``encode_target()`` logic.

        Returns
        -------
        DataFrame with all non-target columns as numeric (no object dtype).
        """
        feature_cols = [c for c in df.columns if c != self.target_col]
        object_cols  = [c for c in feature_cols if df[c].dtype == object]

        if object_cols:
            total_new_nans = 0
            for col in object_cols:
                before = int(df[col].isna().sum())
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after   = int(df[col].isna().sum())
                total_new_nans += max(0, after - before)

            self.logger.warning(
                "Coerced %d object-type column(s) to numeric; %d value(s) could not "
                "be parsed and became NaN (will be filled by handle_missing). "
                "Affected columns: %s",
                len(object_cols), total_new_nans, object_cols,
            )
        else:
            self.logger.info(
                "Numeric coercion: all %d feature columns are already numeric — no action needed.",
                len(feature_cols),
            )

        # Second pass: string literals such as "Infinity" and "inf" become real
        # numeric inf after pd.to_numeric().  clean_data() only caught inf values
        # that were already numeric before coercion, so these new ones slip through.
        # Convert them to NaN here so handle_missing() fills them with training medians,
        # which prevents RobustScaler from receiving infinite values.
        inf_mask = df[feature_cols].isin([np.inf, -np.inf])
        n_inf = int(inf_mask.values.sum())
        if n_inf > 0:
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            self.logger.warning(
                "Post-coercion inf sweep: replaced %d inf/-inf value(s) with NaN "
                "(will be filled by handle_missing with training medians).",
                n_inf,
            )

        return df

    def handle_missing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Fills missing/NaN values with median values calculated exclusively from the training set 
        to prevent data leakage.
        """
        self.logger.info(f"Eksik veriler {'hesaplanıp ' if is_train else ''}dolduruluyor...")
        
        # Exclude target column
        num_cols = df.select_dtypes(include=[np.number]).columns
        if self.target_col in num_cols:
            num_cols = num_cols.drop(self.target_col)
            
        if is_train:
            self.numeric_features_ = list(num_cols)
            for col in self.numeric_features_:
                self.medians_[col] = df[col].median()
                
        # Fill missing values (using the same medians regardless of Train or Test mode)
        for col, median_val in self.medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
                
        return df

    def scale_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Applies RobustScaler which is resilient to outliers.
        Fits only if is_train=True.
        """
        self.logger.info(f"Özellikler RobustScaler ile {'fit edilip ' if is_train else ''}ölçekleniyor...")
        
        features_to_scale = [c for c in self.numeric_features_ if c in df.columns]
        
        if not features_to_scale:
            self.logger.warning("Ölçeklenecek sayısal özellik bulunamadı.")
            return df
            
        if is_train:
            self.scaler_ = RobustScaler()
            df[features_to_scale] = self.scaler_.fit_transform(df[features_to_scale])
        else:
            if self.scaler_ is None:
                raise ValueError("Scaler fit edilmemiş! Önce is_train=True ile çalıştırın.")
            df[features_to_scale] = self.scaler_.transform(df[features_to_scale])
            
        return df

    @abstractmethod
    def handle_multicollinearity(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Dataset-specific multicollinearity and feature elimination rules."""
        pass

    @abstractmethod
    def encode_target(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Dataset-specific target label encoding rules."""
        pass

    def fit_transform(
        self,
        df: pd.DataFrame,
        save_sample: bool = False,
        n_sample_rows: int = 50,
    ) -> pd.DataFrame:
        """Full pipeline execution for the training dataset.

        Args:
            df: Raw training DataFrame.
            save_sample: If True, saves a stratified processed sample to
                         artifacts/samples/processed_sample_{domain}.csv after
                         the pipeline completes. Useful for reporting.
            n_sample_rows: Target number of rows in the saved sample.
        """
        self.logger.info("--- Training Pipeline (fit_transform) Starting ---")
        df = df.copy()
        df = self.clean_data(df)
        df = self._coerce_numeric_features(df)          # guard: object -> numeric before anything else
        df = self.reduce_memory_usage(df)
        df = self.handle_multicollinearity(df, is_train=True)
        df = self.handle_missing(df, is_train=True)     # fills NaN left by coercion with medians
        df = self.scale_features(df, is_train=True)
        df = self.encode_target(df, is_train=True)

        if save_sample:
            self.save_processed_sample(df, n_samples=n_sample_rows)

        self.logger.info("--- Training Pipeline Completed ---")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline execution for Test, Validation or Inference environments."""
        self.logger.info("--- Test/Çıkarım Boru Hattı (transform) Başlıyor ---")
        df = df.copy()
        df = self.clean_data(df)
        df = self._coerce_numeric_features(df)           # guard: object -> numeric before anything else
        df = self.reduce_memory_usage(df)
        df = self.handle_multicollinearity(df, is_train=False)
        df = self.handle_missing(df, is_train=False)     # fills NaN left by coercion with stored medians
        df = self.scale_features(df, is_train=False)
        df = self.encode_target(df, is_train=False)
        self.logger.info("--- Test/Çıkarım Boru Hattı Tamamlandı ---")
        return df

    def save_artifacts(self, filename: str = "preprocessor_artifacts.joblib"):
        """Saves trained state objects to disk for use in production."""
        if not self.artifact_dir.exists():
            self.artifact_dir.mkdir(parents=True)
            
        filepath = self.artifact_dir / filename
        state = {
            'medians': self.medians_,
            'scaler': self.scaler_,
            'label_encoder': self.label_encoder_,
            'dropped_corr_cols': self.dropped_corr_cols_,
            'numeric_features': self.numeric_features_,
            'target_col': self.target_col,
            'task_type': self.task_type
        }
        joblib.dump(state, filepath)
        self.logger.info(f"Model eserleri (Artifacts) başarıyla kaydedildi: {filepath}")

    def load_artifacts(self, filename: str = "preprocessor_artifacts.joblib"):
        """Restores state objects (artifacts) from disk."""
        filepath = self.artifact_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact dosyası bulunamadı: {filepath}")
            
        state = joblib.load(filepath)
        self.medians_ = state['medians']
        self.scaler_ = state['scaler']
        self.label_encoder_ = state['label_encoder']
        self.dropped_corr_cols_ = state['dropped_corr_cols']
        self.numeric_features_ = state['numeric_features']
        self.logger.info(f"Model eserleri (Artifacts) başarıyla yüklendi: {filepath}")

    def save_processed_sample(
        self,
        df: pd.DataFrame,
        n_samples: int = 50,
        output_dir: Optional[Path] = None,
        domain: Optional[str] = None,
    ) -> Path:
        """Save a stratified sample of the processed DataFrame for reporting.

        Performs per-class stratified sampling so that the exported CSV reflects
        the true class distribution rather than being dominated by the majority
        class (e.g. 48 Benign + 2 Attack rows out of 50 would be misleading).

        If a minority class has fewer samples than the equal share, the deficit
        is filled from the majority class so the output always contains
        exactly ``n_samples`` rows (or fewer only when the full dataset is
        smaller than ``n_samples``).

        Args:
            df: Processed DataFrame — output of fit_transform() or transform().
            n_samples: Target total number of rows to export.
            output_dir: Destination directory. Defaults to
                        <project_root>/artifacts/samples/.
            domain: Domain tag used in the filename
                    (e.g. 'cloud' → processed_sample_cloud.csv).
                    Falls back to ``self.domain_name`` if set, else 'unknown'.

        Returns:
            Absolute Path of the saved CSV file.
        """
        # --- Resolve domain label ---
        domain_name = domain or getattr(self, "domain_name", "unknown")

        # --- Resolve output directory ---
        if output_dir is None:
            # Navigate from src/data/preprocessing.py → project root
            project_root = Path(__file__).parents[2]
            output_dir = project_root / "artifacts" / "samples"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Stratified sampling ---
        if self.target_col in df.columns and df[self.target_col].nunique() >= 1:
            class_counts = df[self.target_col].value_counts()
            n_classes = len(class_counts)
            n_per_class = max(1, n_samples // n_classes)

            sample_parts: List[pd.DataFrame] = []
            for cls_val in class_counts.index:
                cls_df = df[df[self.target_col] == cls_val]
                take = min(len(cls_df), n_per_class)
                sample_parts.append(
                    cls_df.sample(n=take, random_state=42)
                )

            sample = pd.concat(sample_parts, ignore_index=True)

            # Top-up: if minority classes were small we may have fewer than
            # n_samples rows — fill the gap from the remaining majority rows.
            if len(sample) < n_samples:
                already_selected = sample.index
                remaining = df.drop(
                    index=df.index[df.index.isin(already_selected)],
                    errors="ignore",
                )
                extra_needed = min(n_samples - len(sample), len(remaining))
                if extra_needed > 0:
                    extra = remaining.sample(n=extra_needed, random_state=42)
                    sample = pd.concat([sample, extra], ignore_index=True)

            # Shuffle so rows are not sorted by class label
            sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

            class_dist = sample[self.target_col].value_counts().to_dict()
            self.logger.info(
                "Stratified sample built — total_rows=%d, class_distribution=%s",
                len(sample), class_dist,
            )
        else:
            self.logger.warning(
                "Target column '%s' not found in DataFrame; "
                "falling back to head(%d) sampling.",
                self.target_col, n_samples,
            )
            sample = df.head(n_samples)

        # --- Persist ---
        output_path = output_dir / f"processed_sample_{domain_name}.csv"
        sample.to_csv(output_path, index=False)
        self.logger.info(
            "Processed sample saved → %s  (rows=%d, features=%d)",
            output_path, len(sample), len(sample.columns),
        )
        return output_path


class CloudPreprocessor(BasePreprocessor):
    """Customized preprocessor for the CIC-IDS2018 (Cloud) dataset."""

    def __init__(self, target_col: str = 'Label', task_type: str = 'binary', artifact_dir: str = '.'):
        super().__init__(target_col=target_col, task_type=task_type, artifact_dir=artifact_dir)
        # Identifies this preprocessor's domain in sample filenames
        self.domain_name: str = "cloud"

    def handle_multicollinearity(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Eliminates zero-variance columns and highly correlated (>=0.98) columns.
        Threshold raised from 0.95 to 0.98 to preserve IAT/packet-size features
        that are critical for cloud traffic classification.
        """
        self.logger.info(f"Cloud çoklu doğrusallık elemesi {'hesaplanıyor' if is_train else 'uygulanıyor'}...")
        
        if is_train:
            self.dropped_corr_cols_ = []
            
            # Select only non-target numerical columns
            num_df = df.select_dtypes(include=[np.number])
            if self.target_col in num_df.columns:
                num_df = num_df.drop(columns=[self.target_col])
                
            # 1. Zero variance check
            variances = num_df.var()
            zero_var_cols = variances[variances == 0].index.tolist()
            self.dropped_corr_cols_.extend(zero_var_cols)
            
            # Continue with the remaining columns
            valid_cols = [c for c in num_df.columns if c not in zero_var_cols]
            
            # 2. High correlation check
            # NaN values can break correlation calculation, so temporarily fill with medians.
            corr_matrix = num_df[valid_cols].fillna(num_df[valid_cols].median()).corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find correlations of 0.98 and above (threshold raised from 0.95)
            to_drop_corr = set()
            for column in upper_tri.columns:
                high_corr_cols = upper_tri.index[upper_tri[column] >= 0.98].tolist()
                for high_corr_col in high_corr_cols:
                    if column not in to_drop_corr and high_corr_col not in to_drop_corr:
                        # Drop the one with lower variance
                        if variances[column] < variances[high_corr_col]:
                            to_drop_corr.add(column)
                        else:
                            to_drop_corr.add(high_corr_col)
            
            self.dropped_corr_cols_.extend(list(to_drop_corr))
            self.logger.info(f"Varyans/Korelasyon nedeni ile düşürülen özellikler: {self.dropped_corr_cols_}")
        
        # Common for Train and Test: drop the identified columns
        cols_to_drop_now = [c for c in self.dropped_corr_cols_ if c in df.columns]
        if cols_to_drop_now:
            df = df.drop(columns=cols_to_drop_now)
            
        return df

    def encode_target(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Encodes the target variable for the Cloud dataset."""
        if self.target_col not in df.columns:
            return df
            
        self.logger.info(f"Hedef değişken ({self.task_type}) kodlanıyor...")
        
        if self.task_type == 'binary':
            # Benign -> 0, everything else -> 1
            df[self.target_col] = df[self.target_col].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
            df[self.target_col] = df[self.target_col].astype(np.int8)
            
        elif self.task_type == 'multiclass':
            if is_train:
                self.label_encoder_ = LabelEncoder()
                df[self.target_col] = self.label_encoder_.fit_transform(df[self.target_col].astype(str))
            else:
                if self.label_encoder_ is None:
                    raise ValueError("LabelEncoder fit edilmemiş!")
                
                # Unseen labels mapping for production safety
                known_classes = set(self.label_encoder_.classes_)
                df[self.target_col] = df[self.target_col].apply(
                    lambda x: x if x in known_classes else 'Unknown'
                )
                
                # We map known labels and fill unknown/unseen ones with -1 to prevent leakage
                mapping = {cls: idx for idx, cls in enumerate(self.label_encoder_.classes_)}
                df[self.target_col] = df[self.target_col].astype(str).map(mapping).fillna(-1).astype(int)
                
        return df


class IoTPreprocessor(BasePreprocessor):
    """Customized preprocessor for the CICIoT2023 (IoT) dataset."""

    def __init__(self, target_col: str = 'label', task_type: str = 'binary', artifact_dir: str = '.'):
        super().__init__(target_col=target_col, task_type=task_type, artifact_dir=artifact_dir)
        # Identifies this preprocessor's domain in sample filenames
        self.domain_name: str = "iot"

    def handle_multicollinearity(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        IoT specific elimination: `Tot size` column is dropped since it has a perfect 1.0 correlation with the `AVG` column.
        """
        col_to_drop = 'Tot size'
        if col_to_drop in df.columns:
            self.logger.info(f"IoT özel kuralı: '{col_to_drop}' sütunu sızıntı/korelasyon nedeniyle siliniyor.")
            df = df.drop(columns=[col_to_drop])
        return df

    def encode_target(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Encodes the target variable for the IoT dataset."""
        if self.target_col not in df.columns:
            return df
            
        self.logger.info(f"Hedef değişken ({self.task_type}) kodlanıyor...")
        
        if self.task_type == 'binary':
            # Any variation containing "benign" -> 0, everything else -> 1
            df[self.target_col] = df[self.target_col].apply(
                lambda x: 0 if 'benign' in str(x).lower() else 1
            )
            df[self.target_col] = df[self.target_col].astype(np.int8)
            
        elif self.task_type == 'multiclass':
            if is_train:
                self.label_encoder_ = LabelEncoder()
                df[self.target_col] = self.label_encoder_.fit_transform(df[self.target_col].astype(str))
            else:
                if self.label_encoder_ is None:
                    raise ValueError("LabelEncoder fit edilmemiş!")
                
                mapping = {cls: idx for idx, cls in enumerate(self.label_encoder_.classes_)}
                df[self.target_col] = df[self.target_col].astype(str).map(mapping).fillna(-1).astype(int)
                
        return df
