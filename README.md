

---

# Brain Tumor Detection

This repository contains a deep learning project for detecting brain tumors using MRI images. The model is built using TensorFlow and Keras, and it classifies MRI images into two categories: 'yes' (tumor) and 'no' (no tumor).

## Dataset
The dataset used in this project is from Kaggle: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

You can install the dependencies using the following command:
```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn
```

## File Description
- `tumorrr.py`: This is the main script that:
  - Loads and preprocesses the data.
  - Splits the data into training and testing sets.
  - Defines and trains the CNN model.
  - Evaluates the model and displays the results.

## Model Architecture
The Convolutional Neural Network (CNN) model consists of:
- Conv2D layers with ReLU activation
- MaxPooling2D layers
- Flatten layer
- Dense layers with ReLU activation
- Dropout layer
- Output layer with Softmax activation

## Usage
1. Clone the repository:
```bash
git clone https://github.com/realmir1/BrainTumor.git
cd BrainTumor
```
2. Run the script:
```bash
python tumorrr.py
```

## Results
- The model is trained for 10 epochs.
- The test accuracy is displayed after training.
- Sample predictions are visualized using Matplotlib.

## Author
- Ali Emir Sertbaş

---

Feel free to modify the README according to your specific needs.
