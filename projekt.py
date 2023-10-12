
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

os.chdir(r'PATH')

directories = ['APPLE', 'BANANA', 'MIXED', 'ORANGE']
labels = [0, 1, 2, 3]

images = []
preprocessed_labels = []

for directory, label in zip(directories, labels):

    file_names = os.listdir(directory)

    for file_name in file_names:

        image_path = os.path.join(directory, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (224, 224))
        normalized_image = resized_image / 255.0
        images.append(normalized_image)
        preprocessed_labels.append(label)

images = np.array(images)
preprocessed_labels = np.array(preprocessed_labels)

mages = np.array(images)
num_samples = images.shape[0]
height = images.shape[1]
width = images.shape[2]
channels = images.shape[3]

images = images.reshape(num_samples, height, width, channels)

X = images
y = preprocessed_labels
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=30, shuffle=True)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1, random_state=30)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')