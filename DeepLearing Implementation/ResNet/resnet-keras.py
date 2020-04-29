
from keras import layers
from keras import models
from keras import backend as K
from keras import initializers

## Ensure Preprocessing of images before using here

image_height = image_width = 224
image_channels = 3

common_filter_size = (3,3)


# Remember the syntax as follows:
# keras.layers.Conv2D(total_filters, kernel_size=(f,f), strides=(s,s), padding='same', activation='relu')
# keras.layers.MaxPool2D -> Used once
# keras.layers.GlobalAveragePooling2D()(x)
# keras.layers.Dense(1000. activation='sigmoid')

def bn(x):
    return layers.BatchNormalization()(x)

def AddResidualBlock(x, each_channel_size, stepchanger = False):
    if stepchanger:
        shortcut = x
        x = layers.Conv2D(each_channel_size, kernel_size=common_filter_size, strides=(2,2), padding='same', activation='relu')(x)
        x = bn(x)
        x = layers.Conv2D(each_channel_size, kernel_size=common_filter_size, strides=(1,1), padding='same', activation='relu')(x)
        x = bn(x)

        # Took some time to fogure out how to zero-pad when increase in channel
        layer = layers.Conv2D(x.shape[3], kernel_size=2, strides=(2,2), use_bias=False, kernel_initializer=initializers.Ones())
        # Not learned!

        layer.trainable = False
        shortcut = layer(shortcut)
        shortcut = bn(shortcut)
        x = layers.add([shortcut,x])
    else:
        shortcut = x
        x = layers.Conv2D(each_channel_size, kernel_size=common_filter_size, strides=(1,1), padding='same', activation='relu')(x)
        x = bn(x)
        x = layers.Conv2D(each_channel_size, kernel_size=common_filter_size, strides=(1,1), padding='same', activation='relu')(x)
        x = bn(x)
        x = layers.add([shortcut, x])
    return x


def NetworkBuilder(_input):
    base_channel_size = 64
    x = layers.Conv2D(base_channel_size, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')(_input)
    x = bn(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    _stepchanger = False
    for i in range(3):
        x = AddResidualBlock(x, base_channel_size, _stepchanger)

    base_channel_size *= 2

    for i in range(4):
        if i==0:
            _stepchanger = True
        else:
            _stepchanger = False
        x = AddResidualBlock(x, base_channel_size, _stepchanger)

    base_channel_size *= 2

    for i in range(6):
        if i==0:
            _stepchanger = True
        else:
            _stepchanger = False
        x = AddResidualBlock(x, base_channel_size, _stepchanger)

    base_channel_size *= 2

    for i in range(3):
        if i==0:
            _stepchanger = True
        else:
            _stepchanger = False
        x = AddResidualBlock(x, base_channel_size, _stepchanger)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax')(x)
    return x

input_tensor = layers.Input(shape=(image_height, image_width, image_channels))
output_tensor = NetworkBuilder(input_tensor)

model = models.Model(inputs=[input_tensor], outputs = [output_tensor])
print(model.summary())
