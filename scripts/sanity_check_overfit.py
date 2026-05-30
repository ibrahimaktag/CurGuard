import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sys

# Ensure src module is in path when running directly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.cloud_specialist import CloudSpecialist
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_overfit_test():
    """
    Sanity Check: Overfit on a single batch.
    Proves that backpropagation flows correctly, the model has capacity to learn,
    and all metrics/logging inside Trainer work without crashing.
    """
    batch_size = 32
    input_dim = 78
    num_classes = 2 # Binary test
    epochs = 40
    
    logger.info("--- BAŞLIYOR: Uçtan Uca Overfit (Sanity Check) Testi ---")
    
    # 1. Generate Mock Data (2 batches of data: 64 samples)
    torch.manual_seed(42)
    X = torch.randn(64, input_dim)
    
    # To make it easy to learn, let's create a clear mathematical pattern:
    # If the sum of features > 0, it's class 1, else class 0
    y = (X.sum(dim=1) > 0).float()
    
    # We will use the same data for train and validation just to test the pipeline thoroughly
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("Mock veri seti (64 örnek) başarıyla oluşturuldu.")
    
    # 2. Initialize Model
    model = CloudSpecialist(input_dim=input_dim, num_classes=num_classes)
    
    # 3. Setup Optimizer
    # Higher learning rate is better for quick overfitting on small data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # 4. Initialize Trainer
    # BCEWithLogitsLoss için pos_weight bir tensör olmalı
    class_weights = torch.tensor([1.0])
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        task_type='binary',
        class_weights=class_weights,
        patience=30, # We want it to overfit, so keep patience high
        model_save_path='artifacts/models/sanity_check_model.pt'
    )
    
    # 5. Train
    logger.info(f"Model {epochs} epoch boyunca mini-veri üzerinde terletiliyor (Hedef: Train Loss -> 0)...")
    best_metrics = trainer.fit(epochs=epochs)
    
    logger.info("--- TAMAMLANDI: Uçtan Uca Overfit Testi ---")
    logger.info(f"Sonuçlar: {best_metrics}")

if __name__ == "__main__":
    run_overfit_test()
