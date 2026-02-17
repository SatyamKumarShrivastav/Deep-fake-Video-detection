# Deep-fake-Video-detection
Deepfake video detection using CNN, Transfer Learning (MobileNetV2), and CNN-LSTM architectures with performance analysis and visualization.

# 🎭 Deepfake Detection Using CNN & CNN-LSTM

A deep learning-based system for detecting Deepfake videos using Convolutional Neural Networks (CNN) and CNN + LSTM architecture.

---

## 📌 Project Overview

Deepfake videos manipulate facial expressions and lip movements using AI techniques.
This project aims to detect such manipulated videos by analyzing spatial and temporal features of video frames.

The system:

* Extracts frames from videos
* Preprocesses images (resize, normalization)
* Applies CNN for spatial feature extraction
* Uses LSTM to capture temporal patterns (for video classification)
* Evaluates performance using multiple metrics and visualizations

---

## 🧠 Models Implemented

### 1️⃣ Basic CNN

Used for frame-level classification.

### 2️⃣ Transfer Learning (MobileNetV2)

* Pretrained on ImageNet
* Faster training
* Better performance on small datasets

### 3️⃣ CNN + LSTM (Video Classification Model)

* CNN extracts frame features
* LSTM learns temporal sequence patterns
* More suitable for video-based deepfake detection

---

## 📂 Project Structure

```
Deepfake-Detection/
│
├── dataset/
│   ├── real/
│   └── fake/
│
├── deepfake_detection.ipynb
├── performance_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab (GPU support)

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/SatyamKumarShrivastav/deepfake-detection.git
cd deepfake-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or run directly in **Google Colab**.

---

## ▶️ How to Run

1. Upload real and fake videos inside:

```
dataset/real/
dataset/fake/
```

2. Run the notebook step-by-step:

   * Load dataset
   * Train model
   * Evaluate performance
   * Generate graphs

---

## 📊 Performance Evaluation

The model performance is analyzed using:

* Accuracy vs Epoch Graph
* Loss vs Epoch Graph
* Confusion Matrix
* Classification Report
* ROC Curve
* Precision-Recall Curve

---

## 📈 Sample Output Metrics

* Training Accuracy
* Validation Accuracy
* Test Accuracy
* Precision
* Recall
* F1-Score

> Note: Performance depends on dataset size. Larger datasets yield better results.

---

## 🔬 Methodology

1. Extract 10 frames per video
2. Resize frames to 128×128
3. Normalize pixel values
4. Train CNN / CNN-LSTM
5. Evaluate using classification metrics

---

## ⚠️ Limitations

* Small dataset may cause overfitting
* No face detection preprocessing (optional improvement)
* Requires larger dataset for real-world deployment

---

## 🚀 Future Improvements

* Face detection using OpenCV
* Fine-tuning pretrained networks
* Data augmentation
* Larger dataset training
* Deployment as Web App (Streamlit / Flask)
* Real-time deepfake detection

---

## 🎓 Academic Purpose

This project is developed for **Knowledge Demonstration** Examination of 6th semester (Academic Year 2025-26).

* AI / Machine Learning coursework
* Deep Learning mini-project
* Research experimentation
* Educational purposes

---

## 📌 Author

**[SATYAM KUMAR SHRIVASTAV]**
B.Tech - Artificial Intelligence & Data Science
[EXCEL ENGINEERING COLLEGE(AUTONOMOUS),Komarapalayam, Tamilnadu]

---

## 📜 License

This project is for educational and research purposes only.

