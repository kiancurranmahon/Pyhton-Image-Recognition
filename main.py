# Python image classification using Tensorflow and Keras
# Author: Kian Curran Mahon

# Trained on provided dataset by Tensorflow

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import random

# Dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Load the MobileNetV2 model pre-trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')
img_paths = list(data_dir.glob('*/*'))  # Grabs all images from all subdirectories

# Select one random image from the dataset
random_image_path = random.choice(img_paths)

# Load and preprocess the randomly selected image
img = tf.keras.preprocessing.image.load_img(random_image_path, target_size=(224, 224))
input_image = tf.keras.preprocessing.image.img_to_array(img)
input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)
input_image = tf.expand_dims(input_image, axis=0)

# Create and store predictions based on inputted image
predictions = model.predict(input_image)
predicted_classes = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]
plt.imshow(img, interpolation='bicubic')
plt.axis('off')
plt.show()
print("Predictions:")
first_prediction = True
for _, class_name, probability in predicted_classes:
    if first_prediction:
        print(f"{class_name}: {probability}")
        first_prediction = False
    else:
        print(f"{class_name}: {probability}")
print()