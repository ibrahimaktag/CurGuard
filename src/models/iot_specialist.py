import torch
import torch.nn as nn
from src.models.base_model import BaseModel
from src.models.attention import MultiHeadSelfAttention

class IoTSpecialist(BaseModel):
    """
    Lightweight CNN + Bi-LSTM + Attention hybrid model for IoT Network Traffic (CICIoT2023).
    Designed to capture high-frequency but low-footprint attack patterns commonly seen in botnets.
    """
    def __init__(self, input_dim: int, num_classes: int = 2, dropout_rate: float = 0.3):
        super(IoTSpecialist, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 1D-CNN Block (Compact Capacity: 32 -> 64 channels)
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        cnn_out_len = input_dim // 4
        lstm_hidden = 64
        
        # Bi-LSTM Block (Compact)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=lstm_hidden, 
            num_layers=1, # Single layer for lightweight execution
            batch_first=True, 
            bidirectional=True
        )
        
        # Attention Block: 4 heads over the Bi-LSTM output (128-dim / 4 = 32 dim per head)
        self.attention = MultiHeadSelfAttention(hidden_dim=lstm_hidden * 2, num_heads=4)
        
        # Classifier Head (Lower Regularization for compact models)
        out_dim = 1 if num_classes <= 2 else num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),  # Dynamic Dropout for hyperparameter optimization
            nn.Linear(64, out_dim)
        )
        
        # Apply specialized weight initialization from BaseModel
        self.initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Features].
        Returns:
            torch.Tensor: Logits of shape [Batch] (binary) or [Batch, num_classes] (multiclass).
        """
        # Reshape for 1D CNN -> [Batch, Channels=1, Seq_Len=Features]
        x = x.unsqueeze(1) 
        
        # Faz A: CNN (Feature Extraction)
        # Output: [Batch, Channels=64, Seq_Len=input_dim//4]
        x = self.cnn_block(x) 
        
        # **CRITICAL DIMENSION TRANSITION**
        # PyTorch LSTM expects [Batch, Seq_Len, Features] when batch_first=True
        # Output: [Batch, Seq_Len=input_dim//4, Channels=64]
        x = x.permute(0, 2, 1) 
        
        # Faz B: Bi-LSTM (Temporal/Sequential Context)
        # Output: [Batch, Seq_Len, 128]
        lstm_out, _ = self.lstm(x) 
        
        # Faz C: Self-Attention (Focusing on critical time steps)
        # Output: [Batch, 128]
        context, attn_weights = self.attention(lstm_out) 
        
        # Faz D: Classification
        logits = self.classifier(context)
        
        # Squeeze if binary classification to match target shapes
        if self.num_classes <= 2:
            logits = logits.squeeze(1)
            
        return logits
