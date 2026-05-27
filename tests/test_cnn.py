import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.cnn import SimpleCNN
from PIL import Image

def test_cnn_forward_pass():
    model = SimpleCNN(num_classes=10)
    # Simulate a batch of 4 grayscale images, sized 28x28 (MNIST format)
    dummy_input = torch.randn(4, 1, 28, 28)
    
    output = model(dummy_input)
    
    # Assert output shape expects 4 items with 10 probabilities each
    assert output.shape == (4, 10)

def test_model_training_step():
    """Verify that the model gradients flow and weights update."""
    model = SimpleCNN(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    dummy_input = torch.randn(2, 1, 28, 28)
    dummy_labels = torch.tensor([3, 7]) # Random dummy labels
    
    # Save a copy of the initial weights of the first layer
    initial_weights = model.conv1_1.weight.clone()
    
    # Forward + Backward
    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    loss.backward()
    optimizer.step()
    
    # Assert weights have been updated (meaning the learning mechanism works)
    assert not torch.equal(initial_weights, model.conv1_1.weight)

def test_app_prediction_logic():
    """Test the inference prediction logic from the app."""
    from app import predict_digit
    
    # Create a dummy blank PIL image (black background)
    dummy_image = Image.new('L', (200, 200), color=0)
    
    # Run the prediction
    predictions = predict_digit(dummy_image)
    
    # Check that we received 10 classes and they are all probability floats
    assert len(predictions) == 10
    assert all(isinstance(val, float) for val in predictions.values())
    assert all(0 <= val <= 1 for val in predictions.values())
    # Sum of probabilities should be very close to 1.0 (softmax behavior)
    assert pytest.approx(sum(predictions.values()), rel=1e-5) == 1.0
