import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stops the training if the monitored metric doesn't improve after a given patience.
    Supports both minimisation (mode='min', e.g. val_loss) and
    maximisation (mode='max', e.g. val_f1).
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0,
                 verbose: bool = False, mode: str = 'max'):
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose
        self.mode      = mode
        self.counter   = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif self._no_improvement(score):
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def _no_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score > self.best_score - self.min_delta
        else:  # 'max'
            return score < self.best_score + self.min_delta


class Trainer:
    """
    Modular PyTorch training loop.
    - Loosely coupled: No external dependency on hyperparameter tuning frameworks (like Optuna).
    - Reverse Class Imbalance: Handles imbalanced datasets via loss weights.
    - Tracks crucial cybersecurity metrics like F1-Score and False Positive Rate (FPR).
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        task_type: str = 'binary',
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        patience: int = 7,
        model_save_path: str = 'artifacts/models/best_model.pt'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.task_type = task_type
        
        # Cihaz (Device) Yönetimi
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_save_path = Path(model_save_path)
        
        # Sınıf Ağırlıkları (Class Weights) Ataması (Ters Sınıf Dengesizliği için)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            
        if self.task_type == 'binary':
            # binary görevler için BCEWithLogitsLoss. pos_weight, pozitif sınıfın (saldırı) ağırlığıdır.
            # Normal trafiği (0) ödüllendirmek için pos_weight < 1 atanmalıdır.
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif self.task_type == 'multiclass':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError("task_type must be 'binary' or 'multiclass'")
            
        # Schedulers & Callbacks: mode='max' çünkü val_f1'i maksimize ediyoruz
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, mode='max')
        
    def train_epoch(self) -> float:
        """Executes a single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_X)
            
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            
        return total_loss / len(self.train_loader.dataset)

    def validate_epoch(self) -> Tuple[float, float, float]:
        """
        Executes a single validation epoch and computes cybersecurity KPI metrics.
        Returns: Tuple of (Validation Loss, F1-Score, False Positive Rate)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                
                if self.task_type == 'binary':
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long()
                else:
                    preds = torch.argmax(logits, dim=1)
                    
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss = total_loss / len(self.val_loader.dataset)
        
        # F1-Score Hesaplaması
        # average='macro' (tüm sınıflara eşit ağırlık) sınıf dengesizliğinde kritik önem taşır.
        average_mode = 'binary' if self.task_type == 'binary' else 'macro'
        f1 = f1_score(all_targets, all_preds, average=average_mode, zero_division=0)
        
        # FPR (False Positive Rate) Hesaplaması
        cm = confusion_matrix(all_targets, all_preds)
        if self.task_type == 'binary':
            if cm.shape == (2,2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            else:
                fpr = 0.0
        else:
            # Multiclass için Macro FPR hesabı
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tp = np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)
            fpr_array = fp / (fp + tn + 1e-9)
            fpr = np.mean(fpr_array)

        return val_loss, f1, fpr

    def fit(
        self,
        epochs: int,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """Execute the main training loop with scheduling, early stopping, and history tracking.

        Args:
            epochs: Maximum number of training epochs.
            epoch_callback: Optional callback invoked after each epoch with
                            (epoch_index, val_loss). Used for Optuna pruning.

        Returns:
            best_metrics: Dict of the best validation metrics observed:
                          {'val_loss', 'val_f1', 'val_fpr'}.
            history: Dict of per-epoch metric lists for learning-curve plots:
                     {'train_loss', 'val_loss', 'val_f1', 'val_fpr'}.
                     Length equals the number of epochs actually trained
                     (may be shorter than `epochs` when early stopping fires).
        """
        logger.info(
            f"Training starting... (mode={self.task_type.upper()}, max_epochs={epochs})"
        )

        best_val_f1  = 0.0           # checkpoint kriteri: F1 maksimize et
        best_metrics: Dict[str, float] = {}

        history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss':   [],
            'val_f1':     [],
            'val_fpr':    [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_f1, val_fpr = self.validate_epoch()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            history['val_fpr'].append(val_fpr)

            # --- LR scheduler: F1'e göre (mode='max') ---
            self.scheduler.step(val_f1)

            logger.info(
                f"Epoch [{epoch:03d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val FPR: {val_fpr:.4f}"
            )

            if epoch_callback is not None:
                epoch_callback(epoch, val_loss)

            # --- Checkpoint: val_f1 artarsa kaydet ---
            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_metrics = {
                    'val_loss': val_loss,
                    'val_f1':   val_f1,
                    'val_fpr':  val_fpr,
                }
                if hasattr(self.model, 'save_model'):
                    self.model.save_model(self.model_save_path)
                else:
                    self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.model_save_path)
                    logger.info(f"New best model saved (val_f1={val_f1:.4f}): {self.model_save_path}")

            # --- Early stopping: F1 yükselmiyor mu? ---
            self.early_stopping(val_f1)
            if self.early_stopping.early_stop:
                logger.warning(
                    f"Early stopping triggered after epoch {epoch}. "
                    f"Best val_f1={best_val_f1:.4f}"
                )
                break

        logger.info(
            f"Training complete. "
            f"Best val_f1={best_metrics.get('val_f1', 0):.4f}, "
            f"best val_loss={best_metrics.get('val_loss', 0):.4f}"
        )
        return best_metrics, history
