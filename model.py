import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ArtifactCNN(nn.Module):
    """
    CNN model for handwritten artifact detection with batch normalization and dropout
    """
    def __init__(self, num_classes=10):
        super(ArtifactCNN, self).__init__()
        
        # Feature extraction layers with batch normalization
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after convolution and pooling
        # For 28x28 input: 28->14->7->3 (with padding adjustments)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path="saved_model/artifact_cnn.pth", num_classes=10, device='cpu'):
    """
    Load trained model from checkpoint
    
    Args:
        model_path (str): Path to the saved model weights
        num_classes (int): Number of artifact classes
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded PyTorch model in evaluation mode
    """
    model = ArtifactCNN(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            # If checkpoint contains state_dict
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        
    except FileNotFoundError:
        print(f"Warning: Model file not found at {model_path}")
        print("Using randomly initialized model for demonstration.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized model for demonstration.")
    
    return model

def predict(model, image_tensor, class_names, device='cpu'):
    """
    Make prediction on a single image
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
        device: Device to run inference on
    
    Returns:
        tuple: (predicted_class, confidence_score, all_probabilities)
    """
    model.eval()
    
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get all class probabilities for visualization
        all_probs = probabilities.squeeze().cpu().numpy()
    
    return predicted_class, confidence_score, all_probs

def get_model_summary(model, input_size=(1, 1, 28, 28)):
    """
    Get model summary including parameter count
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input size: {input_size}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'input_size': input_size
    }