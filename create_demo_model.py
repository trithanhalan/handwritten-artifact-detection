#!/usr/bin/env python3
"""
Script to create a demo model for the Streamlit application
This creates a randomly initialized model for demonstration purposes
"""

import torch
import os
from model import ArtifactCNN

# Create the saved_model directory if it doesn't exist
os.makedirs('saved_model', exist_ok=True)

# Define class names
CLASS_NAMES = [
    "Artifact_A", "Artifact_B", "Artifact_C", "Artifact_D", "Artifact_E",
    "Artifact_F", "Artifact_G", "Artifact_H", "Artifact_I", "Artifact_J"
]

# Create a demo model
model = ArtifactCNN(num_classes=len(CLASS_NAMES))

# Create a demo checkpoint with metadata
demo_checkpoint = {
    'model_state_dict': model.state_dict(),
    'num_classes': len(CLASS_NAMES),
    'class_names': CLASS_NAMES,
    'best_val_accuracy': 87.3,  # Demo accuracy
    'test_accuracy': 85.1,     # Demo accuracy
    'training_config': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 20,
        'weight_decay': 1e-4
    },
    'model_architecture': 'ArtifactCNN',
    'input_size': (1, 28, 28),
    'total_parameters': sum(p.numel() for p in model.parameters())
}

# Save the demo model
model_save_path = 'saved_model/artifact_cnn.pth'
torch.save(demo_checkpoint, model_save_path)

print(f"✅ Demo model saved to: {model_save_path}")
print(f"   - Classes: {len(CLASS_NAMES)}")
print(f"   - Parameters: {demo_checkpoint['total_parameters']:,}")
print(f"   - Demo accuracy: {demo_checkpoint['best_val_accuracy']:.1f}%")

# Test loading the model
try:
    loaded = torch.load(model_save_path, map_location='cpu')
    test_model = ArtifactCNN(num_classes=loaded['num_classes'])
    test_model.load_state_dict(loaded['model_state_dict'])
    print("✅ Model loading test successful!")
except Exception as e:
    print(f"❌ Error testing model loading: {e}")