import argparse
import optuna
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src module is in path when running directly
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.trainer import Trainer
from src.models.cloud_specialist import CloudSpecialist
from src.models.iot_specialist import IoTSpecialist
from src.data.loader import load_dataset_raw, create_dataloader
from src.data.preprocessing import CloudPreprocessor, IoTPreprocessor
from src.training.splits import split_train_val_test_df
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_data(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw data, preprocesses it, and splits it into train/val sets once 
    before starting the Optuna trials to save immense amount of I/O time.
    """
    logger.info(f"{domain.upper()} veri seti yükleniyor ve ön işlemler (preprocessing) başlatılıyor...")
    
    # Optuna hızı için şimdilik limitli veri çekiyoruz (production'da None yapılabilir)
    df_raw = load_dataset_raw(domain=domain, nrows=50000)
    
    # Hedef kolon config'den gelmeli ama şimdilik standart 'Label' diyelim
    target_col = 'Label'
    if target_col not in df_raw.columns and domain == 'iot':
         target_col = 'label' # IoT data might have different target names, assuming 'label' if 'Label' is missing
    
    # Splitting before preprocessing to avoid data leakage
    df_train, df_val, df_test = split_train_val_test_df(
        df_raw, target_column=target_col, test_size=0.1, val_size=0.1, random_state=42
    )
    
    # Preprocessing
    if domain == 'cloud':
        preprocessor = CloudPreprocessor(target_col=target_col)
    else:
        preprocessor = IoTPreprocessor(target_col=target_col)
        
    logger.info("Train seti ön işleme giriyor (fit_transform)...")
    df_train_clean = preprocessor.fit_transform(df_train)
    logger.info("Validation seti ön işleme giriyor (transform)...")
    df_val_clean = preprocessor.transform(df_val)
    
    # Save the target column name to the dataframes for the objective function
    df_train_clean.attrs['target_col'] = target_col
    df_val_clean.attrs['target_col'] = target_col
    
    return df_train_clean, df_val_clean


def calculate_class_weights(df: pd.DataFrame, target_col: str, device: torch.device) -> torch.Tensor:
    """Calculates reverse class imbalance weights for loss functions."""
    # Assuming benign is majority class (e.g., encoded as 0) and attack is minority (encoded as 1)
    # BCEWithLogitsLoss pos_weight = #Negative Samples / #Positive Samples
    class_counts = df[target_col].value_counts()
    
    # If binary
    if len(class_counts) <= 2:
        neg_count = class_counts.get(0, 1)
        pos_count = class_counts.get(1, 1)
        # We want to give higher weight to minority. 
        # If attack (1) is majority, pos_weight will be < 1, punishing misclassification of 0.
        pos_weight = neg_count / pos_count
        return torch.tensor([pos_weight], dtype=torch.float32, device=device)
    else:
        # Multiclass weights
        total = len(df)
        weights = []
        for i in range(len(class_counts)):
            count = class_counts.get(i, 1)
            weights.append(total / (len(class_counts) * count))
        return torch.tensor(weights, dtype=torch.float32, device=device)


def objective(trial: optuna.Trial, args, df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    Optuna objective function for tuning hyperparameters.
    Returns Validation F1-Score to maximize.
    """
    target_col = df_train.attrs.get('target_col', 'Label')
    input_dim = len(df_train.columns) - 1
    num_classes = df_train[target_col].nunique()
    task_type = 'binary' if num_classes <= 2 else 'multiclass'
    
    # 1. Dinamik Arama Uzayı (Search Space) Seçimi
    if args.model == "cloud":
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6)
        model_cls = CloudSpecialist
    else:
        lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        model_cls = IoTSpecialist

    # 2. Veri Yükleme
    train_loader = create_dataloader(
        df_train, target_col=target_col, task_type=task_type, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        df_val, target_col=target_col, task_type=task_type, batch_size=batch_size, shuffle=False
    )
    
    # 3. Model ve Cihaz Ayarları
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(input_dim=input_dim, num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    class_weights = calculate_class_weights(df_train, target_col, device)
    
    # 4. Optuna Pruning (Budama) Callback
    def pruning_callback(epoch: int, val_loss: float):
        # Optuna'ya ara değer (val_loss) raporlanır
        trial.report(val_loss, epoch)
        # MedianPruner kesme kararı verirse TrialPruned hatası fırlatılarak bu deneme öldürülür
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 5. Trainer Döngüsü
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        task_type=task_type,
        class_weights=class_weights,
        device=device,
        patience=5,
        model_save_path=f"artifacts/models/trial_{trial.number}_model.pt"
    )
    
    # Optuna trials use fewer epochs (e.g. 15) for speed.
    # History is discarded here; full history is captured in the Phase-5 run script.
    best_metrics, _ = trainer.fit(epochs=15, epoch_callback=pruning_callback)

    # Metric to maximise: validation F1-Score
    return best_metrics.get('val_f1', 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Network Specialist Models")
    parser.add_argument("--model", choices=["cloud", "iot"], required=True, help="Hedef veri seti ve model dikeyini seçin.")
    parser.add_argument("--trials", type=int, default=25, help="Optuna deneme (trial) sayısı.")
    args = parser.parse_args()
    
    # Optuna başlatılmadan önce veriyi bir kez yükle ve temizle (Zaman kazanımı)
    try:
        df_train, df_val = prepare_data(args.model)
    except Exception as e:
        logger.error(f"Veri yüklenirken hata oluştu (CSV dosyaları eksik olabilir). Hata: {e}")
        logger.info("Test amaçlı (Sanity Check) mock veri kullanılıyor...")
        
        # Test (mock) data generation if actual data fails
        input_dim = 78
        X = np.random.randn(1000, input_dim)
        y = (X.sum(axis=1) > 0).astype(int)
        df_train = pd.DataFrame(X)
        df_train['Label'] = y
        df_val = df_train.copy()
        df_train.attrs['target_col'] = 'Label'
        df_val.attrs['target_col'] = 'Label'

    logger.info(f"--- OPTUNA HYPERPARAMETER TUNING BAŞLIYOR ({args.model.upper()}) ---")
    
    # İlk 5 deneme dokunulmaz (startup), 3 epoch ısınma süresi
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    try:
        study.optimize(lambda trial: objective(trial, args, df_train, df_val), n_trials=args.trials)
    except KeyboardInterrupt:
        logger.warning("Optuna optimizasyonu kullanıcı tarafından durduruldu.")
        
    logger.info("--- OPTUNA OPTİMİZASYONU TAMAMLANDI ---")
    
    best_trial = study.best_trial
    logger.info(f"En İyi Deneme (Trial ID): {best_trial.number}")
    logger.info(f"En İyi F1-Score: {best_trial.value:.4f}")
    logger.info("En İyi Hiperparametreler:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
