# 🧠 Brain MRI Tumor Detection using Keras 🧠



## Table of Contents
- [🔍 Project Overview](#-project-overview)
- [🎯 Objectives](#-objectives)
- [🧰 Technologies Used](#-technologies-used)
  - [Keras](#keras)
  - [Other Libraries](#other-libraries)
- [📁 Project Structure](#-project-structure)
- [🛠️ Setup & Installation](#️-setup--installation)
- [💻 Usage](#-usage)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Making Predictions](#making-predictions)
- [📈 Results](#-results)
- [📚 Resources](#-resources)
- [📝 Contributing](#-contributing)
- [📜 License](#-license)
- [👤 Author](#-author)

---

## 🔍 Project Overview

This project aims to develop a **Convolutional Neural Network (CNN)** model using **Keras** for the **detection of brain tumors** from MRI images. Leveraging deep learning techniques, the model classifies MRI scans into two categories: **'No Tumor'** and **'Tumor Present'**. The high accuracy and efficiency of the model can assist medical professionals in early and accurate diagnosis.

---

## 🎯 Objectives

- **Data Processing**: Efficiently load and preprocess MRI images for model training.
- **Model Development**: Build a robust CNN model using Keras to classify MRI scans.
- **Model Evaluation**: Assess the model's performance using accuracy metrics and visualization.
- **Prediction Visualization**: Display sample predictions to demonstrate model efficacy.

---

## 🧰 Technologies Used

### Keras
**Keras** is a high-level neural networks API, written in Python and capable of running on top of **TensorFlow**, **Theano**, or **CNTK**. It was developed with a focus on enabling fast experimentation and ease of use, making it ideal for building and training deep learning models.

**Key Features of Keras:**
- **User-Friendly API**: Simplifies the creation of complex neural network architectures.
- **Modular**: Composed of configurable modules such as layers, optimizers, and activation functions.
- **Extensible**: Easily extendable for custom components.
- **Backend Flexibility**: Supports multiple backends like TensorFlow, allowing scalability and performance optimization.

In this project, Keras is utilized to construct, compile, and train the CNN model for brain tumor detection.

### Other Libraries
- **Python 3.x**: Primary programming language.
- **OpenCV**: For image processing and manipulation.
- **NumPy**: Numerical operations on large, multi-dimensional arrays.
- **Matplotlib**: Visualization of data and results.
- **Scikit-learn**: Tools for model evaluation and data splitting.

---

## 📁 Project Structure

```
brain-mri-tumor-detection/
├── data/
│   ├── no/
│   └── yes/
├── images/
│   ├── brain_mri.jpg
│   └── results_plot.png
├── notebooks/
│   └── tumor_detection.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

- **data/**: Contains the MRI images categorized into 'no' (no tumor) and 'yes' (tumor present).
- **images/**: Holds images related to the project, such as illustrative figures and result plots.
- **notebooks/**: Jupyter Notebook with the complete code implementation.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Project documentation.
- **LICENSE**: Licensing information.

---

## 🛠️ Setup & Installation

### Prerequisites
- **Python 3.6+**
- **pip** package manager

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/brain-mri-tumor-detection.git
   cd brain-mri-tumor-detection
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**
   - The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).
   - Download and extract the dataset into the `data/` directory following the structure:
     ```
     data/
     ├── no/
     └── yes/
     ```

---

## 💻 Usage

### Data Preparation

The first step involves loading and preprocessing the MRI images. The images are converted to grayscale, resized to a uniform dimension, and normalized.

```python
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define data directory and categories
data_dir = '/kaggle/input/brain-mri-images-for-brain-tumor-detection'
categories = ['no', 'yes']
img_size = 128
data = []
labels = []

# Load and preprocess images
for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (img_size, img_size))
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            print("Error:", e)

# Convert to numpy arrays and normalize
data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
labels = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
```

### Model Architecture

A Sequential CNN model is built with convolutional, pooling, flattening, dense, and dropout layers.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
```

### Training the Model

Compile and train the model using the Adam optimizer and categorical cross-entropy loss.

```python
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
```

### Evaluating the Model

Evaluate the model's performance on the test dataset.

```python
# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Making Predictions

Use the trained model to make predictions on sample images and visualize the results.

```python
import matplotlib.pyplot as plt

# Select sample images
sample_images = X_test[:4]
sample_labels = y_test[:4]
predictions = model.predict(sample_images)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(sample_images[i].reshape(img_size, img_size), cmap='gray')
    true_label = "Yes" if np.argmax(sample_labels[i]) == 1 else "No"
    predicted_label = "Yes" if np.argmax(predictions[i]) == 1 else "No"
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')
plt.show()
```

---

## 📈 Results

### Training & Validation Accuracy

The model's accuracy improved over the training epochs, demonstrating effective learning.

![Model Accuracy](https://github.com/yourusername/brain-mri-tumor-detection/blob/main/images/results_plot.png)

- **Final Training Accuracy**: ~**92%**
- **Final Validation Accuracy**: ~**90%**

### Sample Predictions

The following images showcase the model's ability to correctly classify MRI scans:

![Sample Predictions](https://github.com/yourusername/brain-mri-tumor-detection/blob/main/images/sample_predictions.png)

---

## 📚 Resources

- **Keras Documentation**: [https://keras.io](https://keras.io)
- **TensorFlow Guide for Keras**: [https://www.tensorflow.org/guide/keras](https://www.tensorflow.org/guide/keras)
- **Brain MRI Dataset on Kaggle**: [https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **Scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)

---

## 📝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**
2. **Create a New Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add your message"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

Please ensure your code adheres to the project's coding standards and includes appropriate documentation.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this project as per the license terms.

---

## 👤 Author

---

> **Disclaimer**: This project is intended for educational purposes. Always consult with a medical professional for accurate diagnosis and treatment.

---

## 🤝 Acknowledgements

- **Kaggle** for providing the dataset.
- **TensorFlow & Keras** communities for their invaluable resources.
- **OpenCV** for image processing tools.

---

Feel free to explore the repository, run the code, and contribute to enhancing the model's performance!
