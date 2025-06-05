#  MNIST Digit Classification

This project implements a digit classifier using the MNIST dataset, utilizing deep learning techniques with TensorFlow and Keras.

---

##  Overview

The MNIST dataset is a benchmark dataset containing 70,000 grayscale images of handwritten digits (0 through 9), each of size 28x28 pixels. This notebook walks through:

- Loading and preprocessing the data
- Building and training a neural network using `Keras`
- Evaluating the model performance
- Visualizing results including a confusion matrix heatmap

---

##  Features

- Loading MNIST data using `keras.datasets`
- Visualizing random samples from the dataset
- Creating a `Sequential` deep learning model
- Using ReLU and softmax activations
- Model training and validation
- Plotting training/validation accuracy and loss curves
- Confusion matrix generation and heatmap visualization for performance insight

---

##  Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Seaborn**

---

##  How to Run

1. **Install required libraries** (if running locally):

    ```bash
    pip install tensorflow matplotlib seaborn
    ```

2. **Launch the notebook**:

    ```bash
    jupyter notebook Mnist_classification.ipynb
    ```

---

##  Results

- Achieved **~98% accuracy** on the MNIST test dataset.
- Accuracy and loss plots clearly show model convergence.
- Confusion matrix and heatmap reveal model strengths and areas for improvement.

---

##  Heatmap of Confusion Matrix

A confusion matrix was generated to evaluate class-wise performance. The heatmap (using `seaborn.heatmap`) helps visualize which digits were most commonly confused by the model.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example usage (replace y_true and y_pred with actual labels)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()
```

---

##  Dataset Info

- **Source**: [Keras MNIST Dataset](https://keras.io/api/datasets/mnist/)
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Size**: 28x28 pixels
- **Classes**: 10 (Digits 0–9)

---

##  Model Architecture

### Example Architecture:
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

- **Input Layer**: Flattens 28x28 image into 784-d vector
- **Hidden Layers**: Fully connected (Dense) layers with ReLU
- **Output Layer**: Dense layer with 10 neurons and softmax activation for multi-class classification

---
###  Confusion Matrix Heatmap

To better understand the model's performance across digit classes, here’s the visualized confusion matrix:

![Confusion Matrix Heatmap](confusion_matrix.png)



##  Conclusion

This notebook demonstrates a full pipeline for handwritten digit classification using a neural network. With minimal preprocessing and a simple architecture, it achieves excellent accuracy and provides visual tools for deeper analysis.

---

