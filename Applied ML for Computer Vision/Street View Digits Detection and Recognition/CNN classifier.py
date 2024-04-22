# -*- coding: utf-8 -*-
"""yw3930_Yue Wang_individualProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10rnEMgwRxm_tK1SPV0nw3__CANIVd9YS
"""

import numpy as np
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
import tensorflow as tf

# %matplotlib inline
np.random.seed(34)


def data_preprocess(data_path):
    # Load images and labels
    data_raw = loadmat(data_path)
    images = np.array(data_raw['X'])
    labels = data_raw['y']
    images = np.transpose(images, (3, 0, 1, 2))

    # Convert images to float type and normalize
    # Convert labels to integer type and flatten to 1D array
    images = images.astype('float64') / 255.0
    labels = labels.astype('int64').flatten()

    # One-hot encoding of labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    return images, labels

# Process the training and test data
train_images, train_labels = data_preprocess('/content/train_32x32.mat')
test_images, test_labels = data_preprocess('/content/test_32x32.mat')
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                  test_size=0.15, random_state=34)

# the initial model
keras.backend.clear_session()
model_1 = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same',
                        activation='relu',
                        input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation='softmax')
])

model_1.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print(model_1.summary())

history_1 = model_1.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_val, y_val))

# Visualize train and validation accuracies and losses

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(history_1.history['accuracy'], label='Training Accuracy')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy for Initial Model')
plt.subplot(1, 2, 2)
plt.plot(history_1.history['loss'], label='Training Loss')
plt.plot(history_1.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss for Initial Model')
plt.show()
print("Training completed")

# Evaluate model on test data
test_loss, test_acc = model_1.evaluate(x=test_images, y=test_labels, verbose=0)
print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.format(test_acc, test_loss))


# Final model
keras.backend.clear_session()
model_2 = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same',
                        activation='relu',
                        input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),

    keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),

    keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation='softmax')
])

model_2.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print(model_2.summary())

history_2 = model_2.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val))

# Evaluate train and validation accuracies and losses
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(history_2.history['accuracy'], label='Training Accuracy')
plt.plot(history_2.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy for Final Model')
plt.subplot(1, 2, 2)
plt.plot(history_2.history['loss'], label='Training Loss')
plt.plot(history_2.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss for Final Model')
plt.show()
print("Training completed")

# Calculate accuracy on test data
test_loss, test_acc = model_2.evaluate(x=test_images, y=test_labels, verbose=1)
print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.format(test_acc, test_loss))


test_predictions = model_2.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.ylabel('True Label')
plt.xlabel('Predicted Labes')
plt.show()

print(classification_report(true_labels, test_predictions))