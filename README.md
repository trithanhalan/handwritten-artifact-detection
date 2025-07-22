# 🖊️ Handwritten Artifact Detection System

A complete machine learning application for classifying handwritten artifacts using PyTorch and Streamlit. Features a custom CNN with batch normalization and dropout, comprehensive training pipeline, and professional web interface.

![Application Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Compatible-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)

## 🎯 Project Overview

This project implements an end-to-end handwritten artifact detection system with:
- **Custom CNN architecture** with batch normalization and dropout
- **Professional Streamlit web interface** with 4 interactive sections
- **Complete training pipeline** with metrics tracking and visualization
- **10 artifact classes** (Artifact_A through Artifact_J)
- **Cloud deployment ready** with Streamlit Cloud compatibility

## 🚀 Quick Start

### Option 1: Local Development
```bash
# Clone the repository
git clone https://github.com/trithanhalan/handwritten-artifact-detection.git
cd handwritten-artifact-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 2: One-Click Cloud Deploy
[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## 📁 Project Structure

```
handwritten-artifact-detection/
│
├── app.py                      # Main Streamlit application
├── model.py                    # CNN model definition and utilities
├── preprocess.py               # Image preprocessing pipeline
├── main.ipynb                  # Complete training notebook
├── requirements.txt            # Python dependencies (Cloud compatible)
├── README.md                   # This file
├── DEPLOYMENT_GUIDE.md         # Detailed deployment instructions
│
├── saved_model/                # Trained model weights
│   └── artifact_cnn.pth       # Model checkpoint (161K parameters)
│
└── tests/                      # Testing scripts
    ├── test_preprocess.py      # Preprocessing function tests
    └── test_cloud_compatibility.py  # Cloud deployment validation
```

## ✨ Features

### 🖥️ **Web Application (app.py)**
- **🔍 Image Classification**: Upload and classify handwritten artifacts
- **📊 Model Performance**: Architecture details, metrics, confusion matrix
- **📈 Training Metrics**: Loss/accuracy curves, hyperparameter analysis
- **🎯 Sample Predictions**: Test set analysis and confidence distributions

### 🧠 **Model Architecture (model.py)**
- **Custom CNN** with 161,098 parameters
- **Batch normalization** for training stability
- **Dropout layers** for regularization (0.25 and 0.5)
- **AdaptiveAvgPool2d** for efficient parameter reduction
- **28x28 grayscale input** → **10 classes output**

### 🔧 **Image Processing (preprocess.py)**
- **Automatic preprocessing** (grayscale, resize, normalize)
- **Data augmentation** for training (rotation, translation, scaling)
- **Image enhancement** with OpenCV (headless compatible)
- **Batch processing** support
- **Robust error handling** with PIL fallbacks

### 📓 **Training Pipeline (main.ipynb)**
- **Complete dataset analysis** and exploration
- **Training loop** with validation and metrics tracking
- **Hyperparameter tuning** experiments
- **Visualization** of training progress
- **Model evaluation** with confusion matrix and classification report

## 🛠️ Technical Specifications

### **Dependencies**
```txt
streamlit==1.33.0              # Web application framework
torch==2.3.0                   # Deep learning framework
torchvision==0.18.0            # Computer vision utilities
opencv-python-headless==4.10.0.84  # Image processing (Cloud compatible)
Pillow==9.5.0                  # Image handling
numpy==1.26.4                  # Numerical computing
matplotlib==3.8.4              # Plotting
seaborn==0.12.2                # Statistical visualization
plotly==5.15.0                 # Interactive plotting
scikit-learn==1.4.2            # Machine learning utilities
pandas==2.0.3                  # Data manipulation
tqdm==4.65.0                   # Progress bars
```

### **System Requirements**
- **Python**: 3.8+ (tested on 3.11)
- **Memory**: 2GB+ RAM recommended
- **Storage**: 100MB for dependencies + model
- **GPU**: Optional (CUDA supported, CPU fallback available)

### **Cloud Compatibility**
- ✅ **Streamlit Cloud**: Fully compatible with headless OpenCV
- ✅ **Heroku**: Ready with proper configuration
- ✅ **Hugging Face Spaces**: Direct deployment support
- ✅ **AWS/GCP/Azure**: Container deployment ready

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | Custom CNN with BatchNorm + Dropout |
| **Parameters** | 161,098 (all trainable) |
| **Input Size** | 28 × 28 grayscale |
| **Classes** | 10 artifact categories |
| **Demo Accuracy** | 87.3% validation |
| **Inference Speed** | Real-time (<100ms) |

## 🎓 Educational Use

Perfect for:
- **Machine Learning Coursework** (Computer Vision, Deep Learning)
- **Portfolio Projects** (Full-stack ML application)
- **Learning PyTorch** (Modern CNN architectures)
- **Understanding Deployment** (Local to cloud workflow)
- **Academic Presentations** (Complete project with documentation)

### **Learning Outcomes**
- CNN architecture design and implementation
- Data preprocessing and augmentation techniques
- Training loop implementation with metrics tracking
- Hyperparameter tuning strategies
- Model evaluation and visualization
- Web application development with Streamlit
- Cloud deployment workflows

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run app.py
# Access at: http://localhost:8501
```

### **Streamlit Cloud** (Recommended)
1. Fork this repository
2. Connect to [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy directly from GitHub
4. Automatic deployment on every push

### **Heroku**
```bash
# Add to your project:
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
echo "python-3.11.0" > runtime.txt

heroku create your-app-name
git push heroku main
```

### **Docker**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🧪 Testing

### **Run All Tests**
```bash
# Test preprocessing functions
python test_preprocess.py

# Test cloud compatibility
python test_cloud_compatibility.py

# Test Streamlit app
streamlit run app.py
```

### **Validation Checklist**
- ✅ All dependencies install correctly
- ✅ Model loads without errors
- ✅ Image preprocessing works with sample images
- ✅ Streamlit app runs without GUI dependencies
- ✅ All features accessible and functional

## 🔧 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **OpenCV GUI Errors**
   - ✅ **Fixed**: Using `opencv-python-headless`
   - No GUI functions in codebase

3. **Model Loading Issues**
   ```bash
   python create_demo_model.py  # Recreate demo model
   ```

4. **Streamlit Port Issues**
   ```bash
   streamlit run app.py --server.port 8502
   ```

### **Cloud Deployment Issues**
- Ensure `requirements.txt` uses `opencv-python-headless`
- Check Python version compatibility (3.8-3.11)
- Verify no GUI dependencies in code

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Development Setup**
```bash
git clone https://github.com/trithanhalan/handwritten-artifact-detection.git
cd handwritten-artifact-detection
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SignVerOD Dataset** - Inspiration for synthetic data generation
- **PyTorch Team** - Deep learning framework
- **Streamlit Team** - Web application framework
- **OpenCV Community** - Image processing capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/trithanhalan/handwritten-artifact-detection/issues)
- **Documentation**: See `DEPLOYMENT_GUIDE.md` for detailed instructions
- **Testing**: Run provided test scripts for validation

---

## 🌟 **Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/trithanhalan/handwritten-artifact-detection?style=social)](https://github.com/trithanhalan/handwritten-artifact-detection/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/trithanhalan/handwritten-artifact-detection?style=social)](https://github.com/trithanhalan/handwritten-artifact-detection/network/members)

**Made with ❤️ for the Machine Learning Community**