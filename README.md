# homework2.part2
تمرین دوم بخش دو
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
np.unique(train_labels)
np.max(train_labels)+1
for i in range(10):
  plt.imshow(train_images[i])
  plt.figure()
x_train = train_images.astype('float32')
x_test = test_images.astype('float32')
train_images=[image.reshape(28*28) for image in train_images]
test_images=[image.reshape(28*28) for image in test_images]
x_train = np.array(train_images)
x_test = np.array(test_images)
# 4. Preprocess class labels
y_train = keras.utils.to_categorical(train_labels, num_classes=10)
y_test = keras.utils.to_categorical(test_labels, num_classes=10)
model = Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

opt_rms = keras.optimizers.RMSprop(learning_rate=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt_rms,
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          epochs=25, batch_size=256, validation_data = (x_test, y_test))
