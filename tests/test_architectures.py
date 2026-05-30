import pytest
import torch
from src.models.cloud_specialist import CloudSpecialist
from src.models.iot_specialist import IoTSpecialist

@pytest.fixture
def input_dim():
    """Typical feature dimension for network traffic datasets (e.g. CIC-IDS2018)."""
    return 78

@pytest.fixture
def batch_size():
    return 32

def test_cloud_specialist_binary(input_dim, batch_size):
    """Tests dimension matching and output shape for Cloud binary classification."""
    model = CloudSpecialist(input_dim=input_dim, num_classes=2)
    model.to(model.device)
    
    # Simulate batch of network traffic features
    x = torch.randn(batch_size, input_dim).to(model.device)
    
    # Forward pass
    logits = model(x)
    
    # Binary classification should return shape [Batch]
    assert logits.shape == (batch_size,), f"Expected {(batch_size,)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output contains NaN values"

def test_cloud_specialist_multiclass(input_dim, batch_size):
    """Tests dimension matching and output shape for Cloud multi-class classification."""
    num_classes = 5
    model = CloudSpecialist(input_dim=input_dim, num_classes=num_classes)
    model.to(model.device)
    
    x = torch.randn(batch_size, input_dim).to(model.device)
    logits = model(x)
    
    # Multiclass should return shape [Batch, num_classes]
    assert logits.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output contains NaN values"

def test_iot_specialist_binary(input_dim, batch_size):
    """Tests dimension matching and output shape for IoT lightweight binary classification."""
    model = IoTSpecialist(input_dim=input_dim, num_classes=2)
    model.to(model.device)
    
    x = torch.randn(batch_size, input_dim).to(model.device)
    logits = model(x)
    
    assert logits.shape == (batch_size,), f"Expected {(batch_size,)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output contains NaN values"

def test_iot_specialist_multiclass(input_dim, batch_size):
    """Tests dimension matching and output shape for IoT lightweight multi-class classification."""
    num_classes = 10
    model = IoTSpecialist(input_dim=input_dim, num_classes=num_classes)
    model.to(model.device)
    
    x = torch.randn(batch_size, input_dim).to(model.device)
    logits = model(x)
    
    assert logits.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output contains NaN values"

def test_device_management(input_dim):
    """
    Tests if the BaseModel correctly assigns a valid PyTorch device 
    and handles tensor placement properly.
    """
    model = CloudSpecialist(input_dim=input_dim, num_classes=2)
    device = model.device
    
    assert isinstance(device, torch.device), "model.device is not a valid torch.device"
    
    model.to(device)
    
    # Check if the first parameter is on the correct device
    first_param = next(model.parameters())
    assert first_param.device.type == device.type, f"Parameter device {first_param.device} does not match {device}"
