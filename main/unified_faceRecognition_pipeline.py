# unified_cnn_pipeline.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, SparsePCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import cv2
from PIL import Image
import imghdr
import matplotlib.pyplot as plt
import time

# ===== Preprocessing Functions =====
def preprocess_for_cnn(X_train, X_test):
    X_train_cnn = X_train[..., np.newaxis].astype(np.float32)
    X_test_cnn = X_test[..., np.newaxis].astype(np.float32)
    return X_train_cnn, X_test_cnn

def preprocess_for_mobilenet(X_train, X_test):
    # Resize each image to (128,128) and convert grayscale->RGB by stacking channels
    def resize_and_rgb(X):
        X_resized = []
        for img in X:
            img = img.astype(np.float32)
            img_resized = tf.image.resize(img[..., np.newaxis], (128, 128)).numpy()
            img_rgb = np.repeat(img_resized, 3, axis=-1)  # from (128,128,1) to (128,128,3)
            X_resized.append(img_rgb)
        return np.array(X_resized)
    X_train_rgb = resize_and_rgb(X_train)
    X_test_rgb = resize_and_rgb(X_test)
    return X_train_rgb, X_test_rgb

# ===== Custom CNN Pipeline =====
def create_custom_cnn(input_shape=(96, 96, 1), num_classes=15):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_custom_cnn_pipeline(X_train, X_test, y_train, y_test, epochs=5, batch_size=3):
    X_train_cnn, X_test_cnn = preprocess_for_cnn(X_train, X_test)
    model = create_custom_cnn(input_shape=X_train_cnn.shape[1:], num_classes=len(np.unique(y_train)))
    model.fit(X_train_cnn, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test_cnn, y_test), verbose=2)
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test)
    print(f"\n✅ Custom CNN Test Accuracy: {test_acc:.4f}")
    return model

# ===== MobileNetV2 Pipeline =====
def create_mobilenet_model(input_shape=(128, 128, 3), num_classes=15):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet',
                                                   pooling='avg',
                                                   alpha=0.35)
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_mobilenet_pipeline(X_train, X_test, y_train, y_test, epochs=20, batch_size=8):
    X_train_resized, X_test_resized = preprocess_for_mobilenet(X_train, X_test)
    model = create_mobilenet_model(input_shape=X_train_resized.shape[1:], num_classes=len(np.unique(y_train)))
    model.fit(X_train_resized, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test_resized, y_test), verbose=2)
    test_loss, test_acc = model.evaluate(X_test_resized, y_test, verbose=2)
    print(f"\n✅ MobileNetV2 Test Accuracy: {test_acc:.4f}")
    return model

# ===== PCA + Sparse PCA Functions =====
def load_dataset(dataset='extended'):
    if dataset == 'extended':
        dataset_path = "/kaggle/input/extended-yale-face-b/cropped"
    elif dataset == 'yalefaces':
        dataset_path = "/kaggle/input/yaledata"
    elif dataset == 'oralface':
        dataset_path = "/kaggle/input/oral-face-at-and-t"
    else:
        raise ValueError("Dataset must be either 'extended', 'yalefaces' or 'oralface'")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    images, labels = [], []
    
    if dataset == 'extended':
        subject_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        subject_label_map = {subject: idx for idx, subject in enumerate(subject_dirs)}
        for subject in subject_dirs:
            subject_path = os.path.join(dataset_path, subject)
            label = subject_label_map[subject]
            for file in os.listdir(subject_path):
                file_path = os.path.join(subject_path, file)
                if file_path.endswith(".pgm"):
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face = img[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (96, 96))
                    else:
                        face_resized = cv2.resize(img, (96, 96))
                    img_array = face_resized / 255.0
                    images.append(img_array)
                    labels.append(label)

    elif dataset == 'yalefaces':
        for f in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, f)
            if os.path.isfile(file_path) and imghdr.what(file_path) == 'gif':
                img = Image.open(file_path).convert('L')
                img_cv = np.array(img)
                faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face = img_cv[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (96, 96))
                    img_array = face_resized / 255.0
                else:
                    img_resized = img.resize((96, 96))
                    img_array = np.array(img_resized) / 255.0
                subject_id = f.split('.')[0]
                label = int(subject_id.replace('subject', '')) - 1
                images.append(img_array)
                labels.append(label)

    elif dataset == 'oralface':
        subject_dirs = sorted([f's{i}' for i in range(1, 41)])
        subject_label_map = {subject: idx for idx, subject in enumerate(subject_dirs)}
        for subject in subject_dirs:
            subject_path = os.path.join(dataset_path, subject)
            label = subject_label_map[subject]
            for file in os.listdir(subject_path):
                file_path = os.path.join(subject_path, file)
                if file_path.endswith(".pgm"):
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face = img[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (96, 96))
                    else:
                        face_resized = cv2.resize(img, (96, 96))
                    img_array = face_resized / 255.0
                    images.append(img_array)
                    labels.append(label)

    return np.array(images), np.array(labels)  # RETURN moved outside conditions

def apply_pca(X_train, X_test, method='pca', n_components=50):
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'sparse_pca':
        # SparsePCA does NOT support n_jobs, so no n_jobs param here
       reducer = SparsePCA(n_components=n_components, alpha=1, max_iter=300, tol=1e-4, n_jobs=-1, random_state=42)

    else:
        raise ValueError("method must be 'pca' or 'sparse_pca'")
    
    start = time.time()
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)
    print(f"{method.upper()} transformation time: {time.time() - start:.2f}s")
    return X_train_reduced, X_test_reduced

def run_pca_pipeline(X_train, X_test, y_train, y_test, method='pca', n_components=50):
    # Flatten images for PCA input
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_reduced, X_test_reduced = apply_pca(X_train_flat, X_test_flat, method=method, n_components=n_components)

    # Train classifiers
    classifiers = {
        "SVM": SVC(kernel='linear', random_state=42),
        "LogReg": LogisticRegression(max_iter=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3)
    }
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_reduced, y_train)
        preds = clf.predict(X_test_reduced)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"{method.upper()} {name} Accuracy: {acc:.4f}")
    return results

# ===== Plotting Function =====
# def plot_accuracy_comparison(results_pca, results_spca, acc_custom_cnn, acc_mobilenet):
#     classifiers = ["SVM", "Logistic Regression", "KNN (k=3)", "Custom CNN", "MobileNetV2"]
#     pca_acc = [results_pca.get("SVM", 0), results_pca.get("LogReg", 0), results_pca.get("KNN", 0), acc_custom_cnn, acc_mobilenet]
#     spca_acc = [results_spca.get("SVM", 0), results_spca.get("LogReg", 0), results_spca.get("KNN", 0), acc_custom_cnn, acc_mobilenet]

#     x = np.arange(len(classifiers))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(10,6))
#     rects1 = ax.bar(x - width/2, [a*100 for a in pca_acc], width, label='PCA')
#     rects2 = ax.bar(x + width/2, [a*100 for a in spca_acc], width, label='Sparse PCA')

#     ax.set_ylabel('Accuracy (%)')
#     ax.set_title('Accuracy Comparison')
#     ax.set_xticks(x)
#     ax.set_xticklabels(classifiers)
#     ax.legend()
#     plt.ylim(0, 110)
#     plt.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_comparison(results_pca, results_spca, acc_custom_cnn, acc_mobilenet):
    model_names = [
        'PCA + SVM', 'PCA + LogReg', 'PCA + KNN',
        'Sparse PCA + SVM', 'Sparse PCA + LogReg', 'Sparse PCA + KNN',
        'Custom CNN', 'MobileNetV2'
    ]
    
    accuracies = [
        results_pca.get("SVM", 0),
        results_pca.get("LogReg", 0),
        results_pca.get("KNN", 0),
        results_spca.get("SVM", 0),
        results_spca.get("LogReg", 0),
        results_spca.get("KNN", 0),
        acc_custom_cnn,
        acc_mobilenet
    ]
    
    # Define colors: blue for PCA, green for Sparse PCA, orange for CNNs
    colors = [
        '#1f77b4', '#1f77b4', '#1f77b4',       # PCA models
        '#2ca02c', '#2ca02c', '#2ca02c',       # Sparse PCA models
        '#ff7f0e', '#d62728'                   # CNN and MobileNetV2
    ]
    
    x = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, [a * 100 for a in accuracies], color=colors)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison: PCA, Sparse PCA, and CNN Models')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 110)

    # Annotate each bar with its value
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc*100:.1f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Custom legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label='PCA Models'),
        mpatches.Patch(color='#2ca02c', label='Sparse PCA Models'),
        mpatches.Patch(color='#ff7f0e', label='Custom CNN'),
        mpatches.Patch(color='#d62728', label='MobileNetV2')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()

# ===== Main block =====
if __name__ == "__main__":
    dataset = 'oralface'  # 'oralface', 'extended', or 'yalefaces'
    print(f"Loading {dataset} dataset...")
    images, labels = load_dataset(dataset=dataset)
    print(f"Loaded {len(images)} images, with {len(np.unique(labels))} classes.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Run PCA pipelines
    print("\nRunning PCA pipeline...")
    results_pca = run_pca_pipeline(X_train, X_test, y_train, y_test, method='pca', n_components=50)

    print("\nRunning Sparse PCA pipeline...")
    results_spca = run_pca_pipeline(X_train, X_test, y_train, y_test, method='sparse_pca', n_components=50)

    # Run Custom CNN
    print("\nRunning Custom CNN pipeline...")
    custom_cnn_model = run_custom_cnn_pipeline(X_train, X_test, y_train, y_test, epochs=5, batch_size=3)
    acc_custom_cnn = custom_cnn_model.evaluate(preprocess_for_cnn(X_test, X_test)[0], y_test, verbose=0)[1]

    # Run MobileNetV2
    print("\nRunning MobileNetV2 pipeline...")
    mobilenet_model = run_mobilenet_pipeline(X_train, X_test, y_train, y_test, epochs=20, batch_size=8)
    X_test_mobilenet, _ = preprocess_for_mobilenet(X_test, X_test)
    acc_mobilenet = mobilenet_model.evaluate(X_test_mobilenet, y_test, verbose=0)[1]

    # Plot all accuracies
    plot_accuracy_comparison(results_pca, results_spca, acc_custom_cnn, acc_mobilenet)
