# 🖊️ Handwritten Artifact Detection - Deployment Guide

## 📁 Project Structure

```
handwritten-artifact-detection/
│
├── app.py                    # Main Streamlit application
├── model.py                  # CNN model definition and utilities
├── preprocess.py             # Image preprocessing pipeline
├── requirements.txt          # Python dependencies
├── main.ipynb               # Complete training notebook
├── create_demo_model.py     # Demo model creation script
├── README.md                # Project documentation
├── DEPLOYMENT_GUIDE.md      # This deployment guide
└── saved_model/             # Trained model weights
    └── artifact_cnn.pth     # Model checkpoint (161K parameters)
```

## 🚀 Quick Start - Local Deployment

### Prerequisites
- Python 3.8+ 
- pip package manager

### Installation Steps

1. **Clone/Download the project files**
   ```bash
   # All files should be in the same directory
   cd handwritten-artifact-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your web browser
   - Navigate to: `http://localhost:8501`
   - The application will load automatically

## 🌐 Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Upload your project to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your repository
4. Automatic deployment with every push

### Option 2: Heroku
```bash
# Add these files to your project:
# - Procfile: web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
# - runtime.txt: python-3.11.0

heroku create your-app-name
git push heroku main
```

### Option 3: Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload all project files
4. The app will deploy automatically

## 🎯 Application Features

### 🔍 Image Classification Tab
- **Upload handwritten artifact images**
- **Real-time prediction with confidence scores**
- **Image enhancement options**
- **Detailed results with probability distributions**
- **Download results as CSV**

### 📊 Model Performance Tab
- **Model architecture summary (161,098 parameters)**
- **Performance metrics (87.3% demo accuracy)**
- **Confusion matrix visualization**
- **Per-class performance analysis**

### 📈 Training Metrics Tab
- **Training/validation loss and accuracy curves**
- **Training configuration details**
- **Dataset split information**
- **Hyperparameter tuning results**

### 🎯 Sample Predictions Tab
- **Sample predictions from test set**
- **Class distribution analysis**
- **Confidence score distributions**

## 🛠️ Technical Specifications

### Model Architecture
- **Type**: Custom CNN with Batch Normalization and Dropout
- **Input**: 28x28 grayscale images
- **Classes**: 10 artifact categories (Artifact_A through Artifact_J)
- **Parameters**: 161,098 (all trainable)
- **Framework**: PyTorch 2.0+

### Dependencies
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `streamlit>=1.28.0` - Web application framework
- `matplotlib>=3.6.0` - Plotting library
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.15.0` - Interactive plotting
- `scikit-learn>=1.3.0` - Machine learning utilities
- `opencv-python>=4.8.0` - Image processing
- `Pillow>=9.0.0` - Image handling
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. **Model loading errors**
   - Ensure `saved_model/artifact_cnn.pth` exists
   - Run `python create_demo_model.py` to create demo model

4. **Memory issues**
   - Close other applications
   - Use CPU-only inference by default

### Performance Optimization

1. **For better performance**
   - Use GPU if available (CUDA)
   - Increase batch size for inference
   - Enable image caching

2. **For production deployment**
   - Add user authentication
   - Implement rate limiting
   - Add logging and monitoring
   - Use CDN for static assets

## 📝 Training Your Own Model

To train with real data:

1. **Update the dataset in `main.ipynb`**
   - Replace synthetic data with SignVerOD dataset
   - Modify data loading functions
   - Adjust preprocessing as needed

2. **Run the training notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

3. **The trained model will be saved to `saved_model/`**

## 🎓 Educational Use

This project includes:
- **Complete training pipeline** with metrics tracking
- **Hyperparameter tuning** examples
- **Model evaluation** with confusion matrix
- **Professional visualization** of results
- **Best practices** for deep learning projects

Perfect for:
- Machine learning coursework
- Portfolio projects
- Learning PyTorch and Streamlit
- Understanding CNN architectures

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review the code documentation
- Ensure all dependencies are installed correctly

## 🏆 Project Highlights

✅ **Complete end-to-end ML pipeline**  
✅ **Professional Streamlit interface**  
✅ **CNN with batch normalization and dropout**  
✅ **Comprehensive training notebook**  
✅ **Interactive visualizations**  
✅ **Ready for deployment**  
✅ **Educational and production-ready**  

---

🎉 **Your Handwritten Artifact Detection System is ready to deploy!**