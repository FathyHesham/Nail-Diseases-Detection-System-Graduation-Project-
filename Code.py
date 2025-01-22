# Importing necessary libraries
import itertools
import os  # Required for interacting with the operating system
import cv2  # Required for computer vision tasks, such as image and video processing
import numpy as np  # Required for numerical computations
import matplotlib.pyplot as plt  # Required for visualization
from sklearn.model_selection import train_test_split  # Required for splitting data into training and testing sets
from tensorflow.keras.models import Sequential  # Required for creating a sequential model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization  # Required for defining model layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Required for image data preprocessing
from tensorflow.keras.utils import to_categorical  # Required for one-hot encoding of labels
from tensorflow.keras.optimizers import Adam  # Required for optimizing the model
from keras.callbacks import ModelCheckpoint, EarlyStopping  # Import the necessary callbacks
from tensorflow.keras.applications import VGG19  # Required for importing the VGG19 model
from tensorflow.keras.models import Model  # Required for creating a functional model
from tensorflow.keras.applications.resnet50 import ResNet50  # Required for importing the ResNet50 model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report  # Required for model evaluation


# Define the path to the image directory
data_dir = "/kaggle/input/dataset/Dataset"

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

# Set the image size to 224 x 224
image_size = 224
# Load the images and preprocess them
x = []  # List to store the preprocessed images
y = []  # List to store the corresponding labels

for disease in classes:
    path = os.path.join(data_dir, disease)  # Construct the path to the specific disease class
    class_num = classes.index(disease)  # Get the index of the current disease in the classes list
    for img in os.listdir(path):  # Iterate over the images in the current disease class
        img_arr = cv2.imread(os.path.join(path, img))  # Read the image using OpenCV
        img_arr = cv2.resize(img_arr, (image_size, image_size))  # Resize the image to the desired size
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # Convert the image to RGB color space
        img_arr = img_arr.astype('float32') / 255.  # Normalize the image by scaling pixel values between 0 and 1
        
        x.append(img_arr)  # Add the preprocessed image to the x list
        y.append(class_num)  # Add the corresponding label to the y list


# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)


# Reshape the data
x_train = np.array(x_train).reshape(-1, image_size, image_size, 3)
x_test = np.array(x_test).reshape(-1, image_size, image_size, 3)
# Convert the labels to categorical format
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))


model = Sequential()  # Create a sequential model

# Add convolutional layers with max pooling
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu",
                 input_shape=(image_size, image_size, 3), kernel_regularizer='l2'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.30))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.30))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.30))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer='l2'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())  # Flatten the output of the previous layer

# Add fully connected layers with dropout
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.30))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.30))
model.add(Dense(10, activation="softmax"))  # Output layer with 9 units (assuming 9 classes)


model.compile(optimizer=Adam(learning_rate=0.0001),  # Specify the optimizer with a learning rate
              loss='categorical_crossentropy',  # Specify the loss function for multiclass classification
              metrics=['accuracy'])  # Specify the evaluation metric to monitor during training


model.summary() # make summary


# Define the EarlyStopping callback
earlystopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1)
# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)


# Train the model
history = model.fit(x_train, y_train, batch_size = 256, 
                    epochs = 50, 
                    validation_data=(x_test, y_test), 
                    callbacks = [checkpoint , earlystopping])


# Evaluate the model on training and testing data
train_loss, train_acc = model.evaluate(x_train.reshape(-1, image_size, image_size , 3), y_train, verbose=1)
test_loss, test_acc = model.evaluate(x_test.reshape(-1, image_size, image_size, 3), y_test, verbose=1)

# Print the training and testing accuracy
print("Training accuracy: {:.2f}%".format(train_acc * 100))
print("Testing accuracy: {:.2f}%".format(test_acc * 100))


# Evaluate the model on test data
y_pred = model.predict(np.array(x_test))
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Compute and print accuracy, precision, recall, and f1 score
print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
print("Precision:", precision_score(y_test_classes, y_pred_classes, average='weighted'))
print("Recall:", recall_score(y_test_classes, y_pred_classes, average='weighted'))
print("F1 Score:", f1_score(y_test_classes, y_pred_classes, average='weighted'))

model.save("CNN.h5")  # save the model 

#-----------------------------------------------------------------------------------------------------------------#

# Load the VGG19 model and remove the last layer
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.50)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test))

# Evaluate the model on training and testing data
train_loss, train_acc = model.evaluate(x_train.reshape(-1, image_size, image_size , 3), y_train, verbose=1)
test_loss, test_acc = model.evaluate(x_test.reshape(-1, image_size, image_size, 3), y_test, verbose=1)

# Print the training and testing accuracy
print("Training accuracy: {:.2f}%".format(train_acc * 100))
print("Testing accuracy: {:.2f}%".format(test_acc * 100))

# Evaluate the model on test data
y_pred = model.predict(np.array(x_test))
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print accuracy, precision, recall, and f1 score
print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
print("Precision:", precision_score(y_test_classes, y_pred_classes, average='weighted'))
print("Recall:", recall_score(y_test_classes, y_pred_classes, average='weighted'))
print("F1 Score:", f1_score(y_test_classes, y_pred_classes, average='weighted'))


model.save("VGG19.h5")  # save the model

#---------------------------------------------------------------------------------------------------#

# Load ResNet50 as base model
base_model = ResNet50(include_top=False, input_shape=(image_size, image_size, 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model on training and testing data
train_loss, train_acc = model.evaluate(x_train.reshape(-1, image_size, image_size , 3), y_train, verbose=1)
test_loss, test_acc = model.evaluate(x_test.reshape(-1, image_size, image_size, 3), y_test, verbose=1)

# Print the training and testing accuracy
print("Training accuracy: {:.2f}%".format(train_acc * 100))
print("Testing accuracy: {:.2f}%".format(test_acc * 100))

# Evaluate the model on test data
y_pred = model.predict(np.array(x_test))
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print accuracy, precision, recall, and f1 score
print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
print("Precision:", precision_score(y_test_classes, y_pred_classes, average='weighted'))
print("Recall:", recall_score(y_test_classes, y_pred_classes, average='weighted'))
print("F1 Score:", f1_score(y_test_classes, y_pred_classes, average='weighted'))

plotting_data_dict = history.history  # Get the dictionary containing the training history

# Extract the loss and accuracy values for training and validation data
test_loss = plotting_data_dict['val_loss']
training_loss = plotting_data_dict['loss']
test_accuracy = plotting_data_dict['val_accuracy']
training_accuracy = plotting_data_dict['accuracy']

epochs = range(1, len(test_loss) + 1)  # Generate the x-axis values for epochs

# Plot the training and validation loss
plt.plot(epochs, test_loss, label='Test Loss')
plt.plot(epochs, training_loss, label='Training Loss')
plt.legend()

# Plot the training and validation accuracy
plt.plot(epochs, test_accuracy, label='Test Accuracy')
plt.plot(epochs, training_accuracy, label='Training Accuracy')
plt.legend()

def getConfusionMatrix(self, y_true, y_pred, speaker_labels, title):
    # Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(cm)
    
    # Instantiate Plot Variables
    cmap = plt.cm.Blues  # Color map for the confusion matrix
    title = title  # Plot title
    ticks = np.arange(len(speaker_labels) + 1)  # Ticks for x and y axes
    fmt = 'd'  # Data format
    thresh = cm.max() / 2.  # Threshold for color intensity
    
    # Plot Confusion Matrix
    plt.figure(figsize=(15, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(ticks, speaker_labels, rotation=45)
    plt.yticks(ticks, speaker_labels)
    
    # Add text annotations to the plot
    for (i, j) in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


model.save("ResNet50.h5")  # save the model