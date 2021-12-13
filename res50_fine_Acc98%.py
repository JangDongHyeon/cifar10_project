from os import confstr

import keras

from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10

import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
# from resnetlanb import ResNet18, ResNet34
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2




conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
# conv_base.summary()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)
# input_shape = x_train.shape[1:]
# conv_base = ResNet50(weights='imagenet', include_top=False, input_shape = x_train.shape[1:]))
# conv_base = ResNet50(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])





model = models.Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=20, validation_data=(x_test, y_test))

model.save('my_model.h5')
del model
model = tf.keras.models.load_model('my_model.h5')

model.evaluate(x_test, y_test)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

# plt.figure(figsize=(14, 4))

# plt.subplot(1,2,1)
# plt.plot(epochs, loss_values, 'bo', label='Training Loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(loss_values) + 1)

# plt.subplot(1,2,2)
# plt.plot(epochs, acc, 'bo', label='Training Accuracy', c='orange')
# plt.plot(epochs, val_acc, 'b', label='Validation Accuracy', c='orange')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.figure(figsize=(20,20))
# for i in range(30):
#     plt.subplot(5,6,i+1)
#     # plt.imshow(x_train[i])
    
layer_outputs = [layer.output for layer in conv_base.layers[2:8]]

activation_model = models.Model(inputs=conv_base.input, outputs=layer_outputs)

# img = x_train[30]
# # img = Image.fromarray(img, 'RGB')
# # img.save('outfile.jpg')
# # cv2.imwrite('myImage.png',img)

# # img = image.load_img('outfile.jpg', target_size=(200, 200))
# # img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img, axis=0)
# img_tensor /= 255.

# activations = activation_model.predict(img_tensor)

# first_layer_activation = activations[0]

# plt.matshow(first_layer_activation[0, :, :, 5], cmap='viridis')
# # # plt.show()

# # plt.figure()
# # plt.imshow(x_train[30])
