#!/usr/bin/env python3
"""
Test script to validate Streamlit Cloud compatibility
Runs all critical functions to ensure headless operation
"""

import sys
import traceback
from PIL import Image
import numpy as np

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        import cv2
        print("✅ OpenCV (headless) imported successfully")
        
        from model import ArtifactCNN, load_model, predict
        print("✅ Model functions imported successfully")
        
        from preprocess import preprocess_image, enhance_image_quality
        print("✅ Preprocessing functions imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation and loading"""
    print("\n🔍 Testing model operations...")
    try:
        from model import ArtifactCNN
        model = ArtifactCNN(num_classes=10)
        print("✅ Model creation successful")
        
        # Test model loading
        from model import load_model
        loaded_model = load_model("saved_model/artifact_cnn.pth")
        print("✅ Model loading successful")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def test_image_processing():
    """Test image processing with OpenCV headless"""
    print("\n🔍 Testing image processing...")
    try:
        # Create a test image
        test_image = Image.new('RGB', (64, 64), color='red')
        print("✅ Test image created")
        
        # Test preprocessing
        from preprocess import preprocess_image, enhance_image_quality
        
        # Test basic preprocessing
        tensor = preprocess_image(test_image)
        print(f"✅ Image preprocessing successful: {tensor.shape}")
        
        # Test enhancement (this uses OpenCV)
        enhanced = enhance_image_quality(test_image)
        print(f"✅ Image enhancement successful: {enhanced.size}")
        
        return True
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        traceback.print_exc()
        return False

def test_opencv_headless():
    """Test that only headless OpenCV functions work"""
    print("\n🔍 Testing OpenCV headless compatibility...")
    try:
        import cv2
        
        # Test basic OpenCV operations (headless)
        test_array = np.zeros((50, 50, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Test CLAHE (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        print("✅ OpenCV headless operations successful")
        
        # Verify no GUI functions are available (they should fail)
        try:
            cv2.imshow("test", test_array)  # This should fail in headless
            print("⚠️ WARNING: cv2.imshow is available (not fully headless)")
        except:
            print("✅ cv2.imshow properly disabled (headless confirmed)")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("🚀 STREAMLIT CLOUD COMPATIBILITY TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Operations", test_model_creation), 
        ("Image Processing", test_image_processing),
        ("OpenCV Headless", test_opencv_headless)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - READY FOR STREAMLIT CLOUD!")
        print("✅ Your application is fully compatible with cloud deployment")
    else:
        print("⚠️ SOME TESTS FAILED - CHECK ISSUES ABOVE")
        print("❌ Fix the issues before deploying to Streamlit Cloud")
    
    print("=" * 50)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)