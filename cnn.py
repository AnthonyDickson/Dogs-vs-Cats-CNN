# Convolutional Neural Network
# Achieves ~90% accuracy on the Dogs vs Cats dataset available from
# https://www.kaggle.com/c/dogs-vs-cats

import matplotlib.pyplot as plt
# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Parameters
target_size = (128, 128)
dropout_rate = 0.2
batch_size = 32
epochs = 50

# Initialise the CNN
classifier = Sequential()

# Add convolutional layers
classifier.add(Conv2D(32, (3, 3), input_shape=(*target_size, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#  second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the input
classifier.add(Flatten())

# Add a fully connected layer with dropout
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(dropout_rate))
# Add a second fully connected layer with dropout
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(dropout_rate))

# Add output layer.
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fit the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# Current setup requires the dog and cat images to be in their own folders
# inside of the train and test directories.
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

history = classifier.fit_generator(training_set,
                                   steps_per_epoch=(training_set.n/batch_size),
                                   epochs=epochs,
                                   validation_data=test_set,
                                   validation_steps=(test_set.n/batch_size))

# Summarise history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarise history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Print summary of the model.
print(classifier.summary())

# Load a single image for prediction.
from keras.preprocessing import image
import numpy as np

img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                     target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, 0)

# Print the prediction; 1 is dog and 0 is cat (as in training_set.indicies).
print("dog" if classifier.predict(x)[0] == 1 else "cat")