#!/usr/bin/env python3
"""
Test script to validate preprocess.py functions work without GUI dependencies
"""

import numpy as np
from PIL import Image
import torch

def test_preprocess_functions():
    """Test all preprocessing functions with sample images"""
    print("🔍 Testing preprocess.py functions...")
    
    try:
        from preprocess import preprocess_image, enhance_image_quality, validate_image
        
        # Create test images
        print("\n1. Creating test images...")
        
        # RGB test image
        rgb_image = Image.new('RGB', (64, 64), color=(255, 128, 0))
        print("✅ RGB test image created")
        
        # Grayscale test image
        gray_image = Image.new('L', (64, 64), color=128)
        print("✅ Grayscale test image created")
        
        # Test basic preprocessing
        print("\n2. Testing basic preprocessing...")
        tensor_rgb = preprocess_image(rgb_image)
        tensor_gray = preprocess_image(gray_image)
        
        print(f"✅ RGB preprocessing: {tensor_rgb.shape}")
        print(f"✅ Grayscale preprocessing: {tensor_gray.shape}")
        
        # Verify tensor properties
        assert tensor_rgb.shape == (1, 1, 28, 28), f"Wrong RGB tensor shape: {tensor_rgb.shape}"
        assert tensor_gray.shape == (1, 1, 28, 28), f"Wrong gray tensor shape: {tensor_gray.shape}"
        print("✅ Tensor shapes are correct")
        
        # Test image enhancement (uses OpenCV)
        print("\n3. Testing image enhancement (OpenCV headless)...")
        enhanced_rgb = enhance_image_quality(rgb_image)
        enhanced_gray = enhance_image_quality(gray_image)
        
        print(f"✅ RGB enhancement successful: {enhanced_rgb.size}")
        print(f"✅ Grayscale enhancement successful: {enhanced_gray.size}")
        
        # Test image validation
        print("\n4. Testing image validation...")
        valid_rgb = validate_image(rgb_image)
        valid_gray = validate_image(gray_image)
        invalid_array = np.array([1, 2, 3])  # Invalid image
        invalid_check = validate_image(invalid_array)
        
        assert valid_rgb == True, "RGB image should be valid"
        assert valid_gray == True, "Grayscale image should be valid"
        assert invalid_check == False, "Invalid array should fail validation"
        print("✅ Image validation working correctly")
        
        # Test with numpy array input
        print("\n5. Testing numpy array preprocessing...")
        from preprocess import preprocess_image_array
        
        rgb_array = np.array(rgb_image)
        tensor_from_array = preprocess_image_array(rgb_array)
        print(f"✅ Numpy array preprocessing: {tensor_from_array.shape}")
        
        print("\n🎉 ALL PREPROCESSING TESTS PASSED!")
        print("✅ Functions work without GUI dependencies")
        print("✅ OpenCV headless operations successful")
        print("✅ Ready for Streamlit Cloud deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preprocess_functions()
    if success:
        print("\n🚀 PREPROCESS.PY IS CLOUD-READY!")
    else:
        print("\n⚠️ ISSUES FOUND - CHECK ERRORS ABOVE")