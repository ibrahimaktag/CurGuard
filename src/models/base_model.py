import torch
import torch.nn as nn
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseModel(nn.Module):
    """
    Abstract base class for all deep learning models in the project.
    Provides common utilities like weight initialization, model saving/loading,
    and device (CUDA/CPU) management.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """
        Dynamically detects the best available hardware.
        Returns CUDA if a GPU is available, otherwise falls back to CPU.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model cihaz (device) ayarı: {device}")
        return device
        
    def initialize_weights(self):
        """
        [Expert Best Practice] Centralized weight initialization logic to prevent 
        vanishing/exploding gradients, specifically designed for network traffic anomaly detection.
        
        - Applies Kaiming (He) Normal initialization to Conv1d and Linear layers.
          (Optimal for preserving variance across deep networks utilizing ReLU/LeakyReLU).
        - Applies Orthogonal initialization to LSTM cells.
          (Crucial for preventing exploding gradients in long sequence modeling).
        - Initializes Batch Normalization layers with weights=1 and bias=0.
        """
        logger.info("Ağırlıklar (Weights) özel ilklendirme stratejisiyle (Kaiming/Orthogonal) ayarlanıyor...")
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1 to help long-term memory at the start of training
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
                        
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self) -> int:
        """
        Calculates and returns the total number of trainable parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Eğitilebilir toplam parametre sayısı: {total_params:,}")
        return total_params

    def save_model(self, save_path: Path):
        """
        Saves the model's state dictionary to the specified path.
        Uses .pt or .pth format securely.
        
        Args:
            save_path (Path): Destination path for the model weights.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        logger.info(f"Model ağırlıkları başarıyla kaydedildi: {save_path}")

    def load_model(self, load_path: Path):
        """
        Loads the model's state dictionary from the specified path.
        
        Args:
            load_path (Path): Path to the saved model weights.
        """
        load_path = Path(load_path)
        if not load_path.exists():
            logger.error(f"Model dosyası bulunamadı: {load_path}")
            raise FileNotFoundError(f"No model found at {load_path}")
            
        # map_location ensures compatibility when moving between GPU and CPU
        self.load_state_dict(torch.load(load_path, map_location=self.device))
        self.to(self.device)
        logger.info(f"Model ağırlıkları başarıyla yüklendi: {load_path}")
