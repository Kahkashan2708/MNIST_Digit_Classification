# MNIST Digit Classification 

This project implements a digit classifier using the MNIST dataset and deep learning techniques via TensorFlow/Keras.

##  Overview

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0 to 9), each of size 28x28 pixels. This notebook demonstrates how to:

- Load and preprocess the data
- Build and train a neural network using Keras
- Evaluate model performance
- Visualize predictions and errors

##  Features

- Load MNIST data from `keras.datasets`
- Visualize sample images and their labels
- Build a `Sequential` model using Keras
- Apply activation functions like ReLU and softmax
- Train the model and track accuracy/loss
- Evaluate performance on the test dataset
- Plot confusion matrix and heatmap for analysis

##  Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

## How to Run

1. **Install dependencies** (if running locally):

    ```bash
    pip install tensorflow matplotlib seaborn
    ```

2. **Run the notebook**:

    ```bash
    jupyter notebook Mnist_classification.ipynb
    ```

##  Results

The trained model achieves over **98% accuracy** on the MNIST test dataset. Visualizations such as accuracy/loss curves and confusion matrix heatmaps help understand model behavior.

##  Dataset

- **Source**: [MNIST from Keras](https://keras.io/api/datasets/mnist/)
- **Samples**: 60,000 training images and 10,000 test images
- **Image Size**: 28x28 pixels
- **Classes**: 10 (digits from 0 to 9)

##  Model Architecture (Example)

- **Input Layer**: Flattened 28x28 pixel image (784 neurons)
- **Hidden Layers**: One or more Dense layers with ReLU activation
- **Output Layer**: Dense layer with 10 units (softmax activation)

Example:
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
