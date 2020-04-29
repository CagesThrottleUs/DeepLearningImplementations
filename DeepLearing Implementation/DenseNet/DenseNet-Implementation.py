# This is the implementation of DenseNet-121

from keras import models
from keras import layers
from keras import backend as K
import math

# HyperParameters
growth_rate = k = 32
image_height = image_width = 224
image_channels = 3
theta = 0.5

# Layers Coding
#  layers.Conv2D(filters, kernel_size, strides, padding)

def AddCommonLayers(x):
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.)(x)
    return x

def DenseBlockGenerator(x, total_size):
    for i in range(total_size):
        connection = x
        x = AddCommonLayers(x)
        x = layers.Conv2D(filters=4 * k, kernel_size = (1,1))(x)
        x = AddCommonLayers(x)
        x = layers.Conv2D(filters=k, kernel_size=(3,3), padding='same')(x)
        x = layers.concatenate([connection, x], axis=3) # Important learning here for keras.models all layers should be of keras.layers
    return x

def TransitionBlockGenerator(x):
    x = AddCommonLayers(x)
    x = layers.Conv2D(math.floor(theta * x.shape[3]), kernel_size=(1,1))(x)
    x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    return x

def NetworkBuilder(_input):
    x = layers.Conv2D(filters=2*k, kernel_size=(7,7), strides=(2,2), padding='same')(_input)
    x = AddCommonLayers(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    # Dense Block 1 -  6 layers
    x = DenseBlockGenerator(x, 6)

    # Tranisition Block
    x = TransitionBlockGenerator(x)

    #Dense Block 2 - 12 klayers
    x = DenseBlockGenerator(x,12)

    # Transition Block
    x = TransitionBlockGenerator(x)

    #Dense Block 3 - 24 layers
    x = DenseBlockGenerator(x, 24)

    #Transition Block
    x = TransitionBlockGenerator(x)

    #Dense Block 4 - 16 layers
    x = DenseBlockGenerator(x, 16)

    #Final Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax')(x)

    return x

input_tensor = layers.Input(shape=(image_height, image_width, image_channels))
output_tensor = NetworkBuilder(input_tensor)

model = models.Model(inputs=[input_tensor], outputs = [output_tensor])
print(model.summary())
print(len(model.layers))
