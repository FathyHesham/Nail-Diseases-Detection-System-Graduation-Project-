# Importing necessary libraries
import cv2  # Required for computer vision tasks, such as image and video processing
import os  # Required for interacting with the operating system
import numpy as np  # Required for numerical computations
from keras.preprocessing.image import ImageDataGenerator  # Required for image data preprocessing


# Set the path to your dataset
dataset_path = 'Dataset'
output_path = 'Image_Train'

# Define the list of diseases or classes
classes = ["Acral Lentiginous Melanoma", 
           "Beaus Line", 
           "Blue Finger", 
           "Clubbing", 
           "Healthy Nail", 
           "Koilonychia", 
           "Muehrckes Lines",  
           "Pitting", 
           "Terrys Nail"]


# Set the image size to 244 x 224
image_size = (244, 224)

# Load the images from the dataset
images = []  # List to store the preprocessed images
labels = []  # List to store the corresponding labels

for disease in classes:
    path = os.path.join(dataset_path, disease)  # Construct the path to the specific disease class
    class_num = classes.index(disease)  # Get the index of the current disease in the classes list
    for img in os.listdir(path):  # Iterate over the images in the current disease class
        image = cv2.imread(os.path.join(path, img))  # Read the image using OpenCV
        image = cv2.resize(image, image_size)  # Resize the image to the desired size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        image = cv2.GaussianBlur(image, (15, 15), 0)  # Apply Gaussian blur to the image
        # Perform unsharp masking by adding a weighted combination of the original image and the blurred image
        image = cv2.addWeighted(image, 2, image, -1, 0)
        image = cv2.equalizeHist(image)  # Enhance the image using histogram equalization
        image = image.astype('float32') / 255.  # Normalize the image by scaling pixel values between 0 and 1
        images.append(np.array(image))  # Add the preprocessed image to the images list
        labels.append(class_num)  # Add the corresponding label to the labels list


# Define the data augmentation parameters
train_datagen = ImageDataGenerator(
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        shear_range=0.2, # shear angle in counter-clockwise direction in radians
        zoom_range=0.2, # range for random zoom
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images horizontally
        vertical_flip=True, # randomly flip images vertically
        brightness_range=(0.5,1.5) # randomly adjust brightness of images
)




# Perform data augmentation on the training set
x_augmented = []  # List to store augmented images
y_augmented = []  # List to store corresponding labels

for i in range(len(images)):
    img = images[i]  # Get the original image
    label = labels[i]  # Get the corresponding label

    # Use the train_datagen to generate augmented images and labels
    for x_batch, y_batch in train_datagen.flow(np.expand_dims(img, axis=0), np.expand_dims(label, axis=0),
                                               batch_size=1, save_to_dir=output_path, save_prefix=label,
                                               save_format='jpg'):
        x_augmented.append(x_batch[0])  # Add the augmented image to the list
        y_augmented.append(y_batch[0])  # Add the corresponding label to the list
        break  # Break the loop to generate only one augmented image per original image
