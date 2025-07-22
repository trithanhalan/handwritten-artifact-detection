import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),                   # Resize to 28x28
    transforms.ToTensor(),                         # Convert to tensor [0,1]
    transforms.Normalize((0.5,), (0.5,))          # Normalize to [-1,1]
])

# Data augmentation transform for training
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),                   # Slightly larger for augmentation
    transforms.RandomRotation(15),                 # Random rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Random crop
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    """
    Preprocess a PIL Image for model inference
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension
    """
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def preprocess_image_array(image_array):
    """
    Preprocess a numpy array image for model inference
    
    Args:
        image_array (np.ndarray): Input image as numpy array
    
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension
    """
    # Convert numpy array to PIL Image
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # RGB image
        image = Image.fromarray(image_array.astype(np.uint8))
    elif len(image_array.shape) == 2:
        # Grayscale image
        image = Image.fromarray(image_array.astype(np.uint8), mode='L')
    else:
        raise ValueError(f"Unsupported image shape: {image_array.shape}")
    
    return preprocess_image(image)

def enhance_image_quality(image):
    """
    Apply image enhancement techniques to improve quality
    Streamlit Cloud compatible - uses headless OpenCV operations only
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Enhanced image
    """
    try:
        # Convert to numpy array for OpenCV operations
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # Convert to grayscale for enhancement
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Gaussian blur to reduce noise (headless operation)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(enhanced)
        
        return enhanced_image
        
    except Exception as e:
        # Fallback to PIL-only enhancement if OpenCV fails
        print(f"OpenCV enhancement failed, using PIL fallback: {e}")
        # Simple PIL-based enhancement
        if image.mode != 'L':
            image = image.convert('L')  # Convert to grayscale
        return image

def batch_preprocess_images(image_list, use_augmentation=False):
    """
    Preprocess a batch of images
    
    Args:
        image_list (list): List of PIL Images
        use_augmentation (bool): Whether to apply data augmentation
    
    Returns:
        torch.Tensor: Batch of preprocessed image tensors
    """
    transform_fn = train_transform if use_augmentation else transform
    
    tensors = []
    for image in image_list:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = transform_fn(image)
        tensors.append(tensor)
    
    # Stack into batch
    return torch.stack(tensors)

def denormalize_image(tensor):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
    
    Returns:
        np.ndarray: Denormalized image array
    """
    # Denormalize from [-1,1] to [0,1]
    denormalized = (tensor * 0.5) + 0.5
    
    # Convert to numpy and transpose if needed
    if len(denormalized.shape) == 4:
        # Batch dimension exists
        denormalized = denormalized.squeeze(0)
    
    if len(denormalized.shape) == 3:
        # Channel dimension exists
        denormalized = denormalized.permute(1, 2, 0)
    
    # Convert to numpy and clip values
    img_array = denormalized.numpy()
    img_array = np.clip(img_array, 0, 1)
    
    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    return img_array

def validate_image(image):
    """
    Validate input image format and properties
    
    Args:
        image: Input image (PIL.Image or np.ndarray)
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        if isinstance(image, Image.Image):
            # Check if image has valid dimensions
            width, height = image.size
            if width < 10 or height < 10:
                return False
            return True
        elif isinstance(image, np.ndarray):
            # Check numpy array dimensions
            if len(image.shape) < 2 or len(image.shape) > 3:
                return False
            if image.shape[0] < 10 or image.shape[1] < 10:
                return False
            return True
        else:
            return False
    except:
        return False