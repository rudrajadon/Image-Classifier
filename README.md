# Image Classifier

This repository contains code for training a Convolutional Neural Network (CNN) to classify images into two categories: "AI" and "Real". The model is built using TensorFlow/Keras and trained on a dataset of images stored in Google Drive.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Matplotlib

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rudrajadon/Image-Classifier.git
   cd Image-Classifier
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Dataset

The dataset is organized into three directories:

- `Train_Images`: Contains training images for model training.
- `Validation_Images`: Contains validation images for evaluating model performance during training.
- `Test_Images`: Contains images for making predictions after model training.

### Training

1. Mount Google Drive and specify the paths to training and validation directories.
2. Define and compile the CNN model using `tf.keras`.
3. Train the model using `model.fit()` function.
4. Evaluate the model's performance on validation data.

### Making Predictions

1. Load test images from the `Test_Images` directory.
2. Use the trained model to predict the class of each image.
3. Display the predicted class ("AI" or "Real") along with the prediction probability.

### Saving the Model

The trained model is saved as `trained_model.h5` in Google Drive.

### Example Usage

```python
# Example code snippet to load and use the trained model for prediction
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('path/to/trained_model.h5')

# Load an image for prediction
img_path = 'path/to/test_image.jpg'
img = image.load_img(img_path, target_size=(200, 200))
plt.imshow(img)
plt.show()

# Preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

# Predict the class probabilities
predictions = model.predict(x)

# Display the prediction result
if predictions[0] < 0.5:
    print("Prediction: AI")
else:
    print("Prediction: Real")
```

### Download the Model

You can download the trained model from [Google Drive](https://drive.google.com/file/d/1iLxWlZa2TU-i3TJLiWZtPdIkNY2rIe9p/view?usp=sharing).

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- This project was created as part of learning TensorFlow/Keras for image classification.
