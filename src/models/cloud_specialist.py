import torch
import torch.nn as nn
from src.models.base_model import BaseModel
from src.models.attention import MultiHeadSelfAttention

class CloudSpecialist(BaseModel):
    """
    High-capacity CNN + Bi-LSTM + Attention hybrid model for Cloud Network Traffic (CIC-IDS2018).
    Designed to capture complex infiltration and highly variable attack patterns.
    """
    def __init__(self, input_dim: int, num_classes: int = 2, dropout_rate: float = 0.5):
        super(CloudSpecialist, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 1D-CNN Block (High Capacity: 64 -> 128 channels)
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate sequence length after CNN (each MaxPool reduces dim by half)
        # Note: In PyTorch, input_dim // 4 might be slightly off if input_dim isn't divisible by 4 perfectly.
        # But MaxPool1d(2) handles it.
        cnn_out_len = input_dim // 4 
        lstm_hidden = 128
        
        # Bi-LSTM Block
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention Block: 4 heads over the Bi-LSTM output (256-dim / 4 = 64 dim per head)
        self.attention = MultiHeadSelfAttention(hidden_dim=lstm_hidden * 2, num_heads=4)
        
        # Classifier Head (High Regularization for Cloud attacks)
        # Binary Classification uses output dimension 1
        out_dim = 1 if num_classes <= 2 else num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),  # Dynamic Dropout for hyperparameter optimization
            nn.Linear(128, out_dim)
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
        # Output: [Batch, Channels=128, Seq_Len=input_dim//4]
        x = self.cnn_block(x) 
        
        # **CRITICAL DIMENSION TRANSITION**
        # PyTorch LSTM expects [Batch, Seq_Len, Features] when batch_first=True
        # Output: [Batch, Seq_Len=input_dim//4, Channels=128]
        x = x.permute(0, 2, 1) 
        
        # Faz B: Bi-LSTM (Temporal/Sequential Context)
        # Output: [Batch, Seq_Len, 256]
        lstm_out, _ = self.lstm(x) 
        
        # Faz C: Self-Attention (Focusing on critical time steps)
        # Output: [Batch, 256]
        context, attn_weights = self.attention(lstm_out) 
        
        # Faz D: Classification
        logits = self.classifier(context)
        
        # Squeeze if binary classification to match target shapes
        if self.num_classes <= 2:
            logits = logits.squeeze(1)
            
        return logits
