# Predictive system code

import numpy as np
import matplotlib.pyplot as plt
import cv2


input_image_path = 'path of image'  # paste path of your image to be predicted

# Load and process image
input_image = cv2.imread(input_image_path)
grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
input_image_resized = cv2.resize(grayscale, (28, 28))
input_image_resized = input_image_resized / 255.0

# Prepare image for model
image_reshape = np.reshape(input_image_resized, [1, 28, 28, 1])

# Predict using your trained model
input_prediction = model.predict(image_reshape)
input_pred_label = np.argmax(input_prediction)

# Print prediction
print("The handwritten digit is recognized as:", input_pred_label)

# Show the image with prediction
plt.imshow(input_image_resized, cmap='gray')
plt.title(f"Predicted Digit: {input_pred_label}")
plt.axis('off')
plt.show()