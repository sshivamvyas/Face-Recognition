# Face-Recognition
Eigenfaces vs. Deep Learning for Face Recognition.

This project explores and compares two different approaches to face recognition: traditional linear methods using Eigenfaces (PCA/Sparse PCA) and modern lightweight deep learning models (CNN and MobileNetV2). The objective was to assess how each performs in terms of accuracy, efficiency, and robustness against noise like lighting and pose variations. We applied PCA and Sparse PCA to reduce dimensionality, followed by classical classifiers such as SVM, Logistic Regression, and KNN. Simultaneously, we trained a 2-layer custom CNN and a pre-trained MobileNetV2 to classify facial images. Experiments were conducted on four datasets: Georgia Tech, Oral Face, Yale Faces, and Extended Yale B. Our results show that while deep learning models outperform in general, PCA-based methods still offer competitive performance with significantly lower computational cost.



```markdown
# Unified Face Recognition Pipeline

A unified pipeline for evaluating traditional and deep learning-based face recognition models on multiple datasets.
This project compares PCA, Sparse PCA (with SVM, Logistic Regression, KNN) against modern CNN-based methods (Custom CNN, MobileNetV2)
on face datasets like ORL, Yale, Extended Yale B, and Georgia Tech.



##  Highlights

-  **Dataset Support**: ORL, Yale Faces, Extended Yale B
-  **Techniques**: PCA, SparsePCA, CNN, MobileNetV2
-  **Classifiers**: SVM, Logistic Regression, KNN
-  **Modular Design**: Easy to switch datasets or models
-  **Accuracy Comparison Plot**: Visualize performance

---

## 📂 Dataset Options

| Dataset        | Variations                                          |
|----------------|-----------------------------------------------------|
| ORL Face       | Pose, lighting, and expression changes              |
| Yale Faces     | Lighting, expression variations (happy, sad, etc.) |
| Extended Yale B| Illumination and shadow variations                 |
| Georgia Tech   | Varying lighting and complex backgrounds           |

📁 Make sure your dataset directory is structured as follows (example for ORL):



/kaggle/input/oral-face-at-and-t/s1/1.pgm
/kaggle/input/yaledata/subject01.happy.gif
/kaggle/input/extended-yale-face-b/cropped/yaleB01/...



---

##  Models & Pipelines

### 1. PCA / Sparse PCA Pipelines
- Images are flattened and transformed using PCA or SparsePCA
- Classification using:
  - **Support Vector Machines (SVM)**
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**

### 2. Custom CNN
- Lightweight CNN with two Conv layers
- Designed for 96×96 grayscale face inputs

### 3. MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned using average pooling and 2 dense layers
- Input resized to 128×128 RGB

---

## 🛠️ System Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- PIL / Pillow
- NumPy
- Matplotlib

Tested on:
- **Kaggle GPU Environment**
- **GPU Type**: Tesla P100

---

## 🚀 How to Use

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/unified-face-recognition.git
cd unified-face-recognition
````

### Step 2: Setup Environment

```bash
pip install -r requirements.txt
```

> (On Kaggle, most dependencies are pre-installed)

### Step 3: Choose Dataset and Run

In `unified_faceRecognition_pipeline.py`, change:

```python
dataset = 'oralface'  # Options: 'oralface', 'extended', 'yalefaces', 'georgia'
```

Then execute:

```bash
python unified_faceRecognition_pipeline.py
```

---




## 📈 Results Snapshot (Example: ORL Dataset)

| Method      | SVM | LogReg | KNN |
| ----------- | --- | ------ | --- |
| PCA         | 67% | 59%    | 64% |
| Sparse PCA  | 66% | 56%    | 70% |

| Model       | Acc.|
| ----------- | --- |
| Custom CNN  | 85% | 
| MobileNetV2 | 92% |

---

##  Future Improvements

* Add more datasets (LFW, CelebA)
* Hyperparameter tuning support
* Add visualization of eigenfaces and filters
* Cross-dataset evaluation metrics

---

##  Author

**Shivam Vyas**
M.Tech Data Science & AI @ IIT Madras
Email: \[[sshivamvyas@gmail.com](mailto:sshivamvyas@gmail.com)]


