## In the original Model /references/ResNeXt.pdf Figure 3(c) was giving me Resource Exhausted Error

from keras.models import Model
from keras import layers
from keras import initializers

#hyper-parameters
image_height = image_width = 224
image_channels = 3
cardinality = 32

def AddCommonLayers(x):
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.)(x)
    return x

def ResidualBlockGenerator(x, channels_in, channels_out, stepChanger=False):
    #stepChanger is for reducing size of the feature map like 56 x 56 to 28 x 28
    if stepChanger:
        shortcut = x
        group_size = (channels_in)//(cardinality)
        groups =[]
        for i in range(cardinality):
            groupsElements = layers.Conv2D(group_size, kernel_size=(1,1), strides=(2,2), padding='same')(x)
            groupsElements = AddCommonLayers(groupsElements)
            groupsElements = layers.Conv2D(group_size, kernel_size=(3,3), padding='same')(groupsElements)
            groupsElements = AddCommonLayers(groupsElements)
            groups.append(groupsElements)
        x = layers.concatenate(groups)
        x = layers.Conv2D(channels_out, kernel_size=(1,1))(x)
        x = layers.BatchNormalization()(x)

        layer = layers.Conv2D(channels_out, kernel_size=(2,2), strides=(2,2), use_bias=False, kernel_initializer=initializers.Ones())
        layer.trainable = False
        shortcut = layer(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([shortcut, x])
        x = layers.LeakyReLU(alpha=0.)(x)
    else:
        shortcut = x
        group_size = (channels_in)//(cardinality)
        groups=[]
        for i in range(cardinality):
            groupsElements = layers.Conv2D(group_size, kernel_size=(1,1))(x)
            groupsElements = AddCommonLayers(groupsElements)
            groupsElements = layers.Conv2D(group_size, kernel_size=(3,3), padding='same')(groupsElements)
            groupsElements = AddCommonLayers(groupsElements)
            groups.append(groupsElements)
        x = layers.concatenate(groups)
        x = layers.Conv2D(channels_out, kernel_size=(1,1))(x)
        x = layers.BatchNormalization()(x)

        if shortcut.shape[3] != x.shape[3]:
            layer = layers.Conv2D(channels_out, kernel_size=(1,1), use_bias=False, kernel_initializer = initializers.Ones())
            layer.trainable = False
            shortcut = layer(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([shortcut, x])
        x = layers.LeakyReLU(alpha=0.)(x)
    return x

def NetworkBuilder(_input):
    x = layers.Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same')(_input)
    x = AddCommonLayers(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    # Residual Blocks Section
    for i in range(3):
        x = ResidualBlockGenerator(x, 128, 256, stepChanger=False) # Any number should work

    for i in range(4):
        if i==0:
            _stepChanger = True
        else:
            _stepChanger = False
        x = ResidualBlockGenerator(x, 256, 512, stepChanger = _stepChanger)

    for i in range(6):
        if i==0:
            _stepChanger = True
        else:
            _stepChanger = False
        x = ResidualBlockGenerator(x, 512, 1024, stepChanger = _stepChanger)

    for i in range(3):
        if i == 0:
            _stepChanger = True
        else:
            _stepChanger = False
        x = ResidualBlockGenerator(x, 1024, 2048, stepChanger = _stepChanger)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax')(x)

    return x

input_tensor = layers.Input(shape=(image_height, image_width, image_channels))
output_tensor = NetworkBuilder(input_tensor)

model = Model(inputs=[input_tensor], outputs = [output_tensor])
print(model.summary())
