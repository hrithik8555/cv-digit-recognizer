import pytest
import torch
from src.models.cnn import SimpleCNN

def test_cnn_forward_pass():
    model = SimpleCNN(num_classes=10)
    # Simulate a batch of 4 grayscale images, sized 28x28 (MNIST format)
    dummy_input = torch.randn(4, 1, 28, 28)
    
    output = model(dummy_input)
    
    # Assert output shape expects 4 items with 10 probabilities each
    assert output.shape == (4, 10)
