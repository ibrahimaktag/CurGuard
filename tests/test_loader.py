import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.loader import NetworkDataset, create_dataloader

@pytest.fixture
def mock_preprocessed_df():
    """
    Creates a mock preprocessed pandas DataFrame for testing loader functions.
    This simulates data that has already passed through CloudPreprocessor or IoTPreprocessor.
    """
    return pd.DataFrame({
        'feature_1': [0.1, 0.2, 0.3, 0.4],
        'feature_2': [1.0, 2.0, 3.0, 4.0],
        'target_col': [0, 1, 0, 2] # Simulating up to 3 classes (0, 1, 2)
    })

def test_network_dataset_binary(mock_preprocessed_df):
    """
    Tests NetworkDataset initialization, tensor conversion, and types for a Binary classification task.
    In binary tasks, labels should be returned as float32 for BCEWithLogitsLoss compatibility.
    """
    dataset = NetworkDataset(df=mock_preprocessed_df, target_col='target_col', task_type='binary')
    
    # Assert length
    assert len(dataset) == 4
    
    # Fetch first item
    X, y = dataset[0]
    
    # Assert PyTorch Tensor instance
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    
    # Assert Tensor datatypes
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32  # Binary expects float32
    
    # Assert Tensor shapes
    assert X.shape == (2,) # 2 features
    assert y.shape == ()   # Scalar

def test_network_dataset_multiclass(mock_preprocessed_df):
    """
    Tests NetworkDataset tensor types for a Multiclass classification task.
    In multiclass tasks, labels should be returned as long (int64) for CrossEntropyLoss compatibility.
    """
    dataset = NetworkDataset(df=mock_preprocessed_df, target_col='target_col', task_type='multiclass')
    
    # Fetch first item
    X, y = dataset[0]
    
    # Assert Tensor datatypes
    assert X.dtype == torch.float32
    assert y.dtype == torch.long  # Multiclass expects long
    
def test_create_dataloader(mock_preprocessed_df):
    """
    Tests the create_dataloader wrapper function to ensure it properly initializes
    a PyTorch DataLoader with the correct configurations.
    """
    batch_size = 2
    loader = create_dataloader(
        df=mock_preprocessed_df,
        target_col='target_col',
        task_type='binary',
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Assert proper instantiation
    assert isinstance(loader, DataLoader)
    
    # Fetch a batch
    batch_X, batch_y = next(iter(loader))
    
    # Assert batch shapes
    assert batch_X.shape == (batch_size, 2)
    assert batch_y.shape == (batch_size,)
    
    # Assert batch data types
    assert batch_X.dtype == torch.float32
    assert batch_y.dtype == torch.float32
