import math
import keras.layers as layers
import keras.activations as activations
from keras.initializers import VarianceScaling

# Stage 1
def stem(x, filters):
    """
    Stage 1 of Efficient Net

    Args:
        x: input layer, with scaled and normalized data
        filters: int, number of filters to use in Conv2D layer

    Returns:
        output layer
    """
    x = layers.Conv2D(filters=filters,
                      kernel_size=(3, 3),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=VarianceScaling(scale=2.0),
                      name='Stem_Conv')(x)

    x = layers.BatchNormalization(name='Stem_BN')(x)

    x = layers.Activation(activation=activations.swish,
                          name='Stem_Activation')(x)

    return x


# Mobile Inverted Bottleneck
def MBConv(x, in_channels, out_channels, expansion_factor, kernel_size, strides, name):
    """
    Mobile Inverted Block

    Args:
        x: input layer
        in_channels: int, number of input channels without expansion
        out_channels: int, number of output channels
        expansion_factor: int, factor multiplied by in_channels to get number of filters in Expansion phase
        kernel_size: (int, int), kernel size for DepthwiseConv2D layer
        strides: (int, int), stride for DepthwiseConv2D layer
        name: string, prefix for names of all layers

    Returns:
        output layer
    """
    if expansion_factor > 1:
        x = layers.Conv2D(filters=expansion_factor*in_channels,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          use_bias=False,
                          kernel_initializer=VarianceScaling(scale=2.0),
                          name=name+'_Expand_Conv')(x)

        x = layers.BatchNormalization(name=name+'_Expand_BN')(x)

        x = layers.Activation(activation=activations.swish,
                              name=name+'_Expand_Activation')(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=VarianceScaling(scale=2.0),
                               name=name+'_DWConv')(x)

    x = layers.BatchNormalization(name=name+'_BN')(x)

    x = layers.Activation(activation=activations.swish,
                          name=name+'_Activation')(x)

    # Squeeze and Excitation Optimization
    y = layers.GlobalAveragePooling2D(keepdims=True,
                                      name=name+'_Squeeze')(x)

    y = layers.Conv2D(filters=in_channels//4,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      activation=activations.swish,
                      kernel_initializer=VarianceScaling(scale=2.0),
                      name=name+'_Reduce')(y)

    y = layers.Conv2D(filters=expansion_factor*in_channels,
                      kernel_size=(1, 1),
                      strides=(1, 1), padding='same',
                      activation=activations.sigmoid,
                      kernel_initializer=VarianceScaling(scale=2.0),
                      name=name+'_Expand')(y)

    x = layers.Multiply(name=name+'_Excite')([x, y])

    x = layers.Conv2D(filters=out_channels,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=VarianceScaling(scale=2.0),
                      name=name+'_Project_Conv')(x)

    x = layers.BatchNormalization(name=name+'_Project_BN')(x)

    return x


# Stage 2 to 8
def Block(x, depth, in_channels, out_channels, expansion_factor, kernel_size, strides, name):
    """
    Sequentially Stacked Mobile Inverted Blocks

    Args:
        x: input layer
        in_channels: int, number of input channels for first Block without expansion
        out_channels: int, number of output channels
        expansion_factor: int, factor multiplied by in_channels to get number of filters in Expansion phase of all Blocks
        kernel_size: (int, int), kernel size for all Blocks
        strides: (int, int), stride for all Blocks
        name: string, prefix for names of all Blocks

    Returns:
        output layer

    """
    x = MBConv(x,
               in_channels=in_channels,
               out_channels=out_channels,
               expansion_factor=expansion_factor,
               kernel_size=kernel_size,
               strides=strides,
               name=name+'_1')

    prev_x = x
    for i in range(1, depth):
        x = MBConv(x,
                   in_channels=out_channels,
                   out_channels=out_channels,
                   expansion_factor=expansion_factor,
                   kernel_size=kernel_size,
                   strides=(1, 1),
                   name=f'{name}_{i+1}')

        x = layers.Dropout(rate=0.025,
                           name=f'{name}_{i+1}_DO')(x)

        x = layers.Add(name=f'{name}_{i+1}_Add')([prev_x, x])

        prev_x = x

    return x


# Stage 9
def top(x, filters):
    """
    Stage 9 of Efficient Net

    Args:
        x: input layer
        filters: int, number of filters to use in Conv2D layer

    Returns:
        output layer
    """
    x = layers.Conv2D(filters=filters,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=VarianceScaling(scale=2.0),
                      name='Top_Conv')(x)

    x = layers.BatchNormalization(name='Top_BN')(x)

    x = layers.Activation(activation=activations.swish,
                          name='Top_Activation')(x)

    x = layers.GlobalMaxPool2D(name='Top_MPool')(x)
    return x


def efficient_net(x, depth_factor=1, width_factor=1):
    """
    Efficient Net

    Args:
        x: input layer, with scaled and normalized data
        depth factor: float >= 1.0, Factor by which depth of Efficient Net B0 is increased
        width_factor: float >= 1.0, Factor by which width of Efficient Net B0 is increased

    Returns:
        output layer

    """
    x = stem(x, math.ceil(width_factor * 32/8)*8)

    x = Block(x,
              depth=math.ceil(depth_factor * 1),
              in_channels=math.ceil(width_factor * 32/8) * 8,
              out_channels=math.ceil(width_factor * 16/8) * 8,
              expansion_factor=1,
              kernel_size=(3, 3),
              strides=(1, 1),
              name='Block_1')

    x = Block(x,
              depth=math.ceil(depth_factor * 2),
              in_channels=math.ceil(width_factor * 16/8) * 8,
              out_channels=math.ceil(width_factor * 24/8) * 8,
              expansion_factor=6,
              kernel_size=(3, 3),
              strides=(2, 2),
              name='Block_2')

    x = Block(x,
              depth=math.ceil(depth_factor * 2),
              in_channels=math.ceil(width_factor * 24/8) * 8,
              out_channels=math.ceil(width_factor * 40/8) * 8,
              expansion_factor=6,
              kernel_size=(5, 5),
              strides=(2, 2),
              name='Block_3')

    x = Block(x,
              depth=math.ceil(depth_factor * 3),
              in_channels=math.ceil(width_factor * 40/8) * 8,
              out_channels=math.ceil(width_factor * 80/8) * 8,
              expansion_factor=6,
              kernel_size=(3, 3),
              strides=(2, 2),
              name='Block_4')

    x = Block(x,
              depth=math.ceil(depth_factor * 3),
              in_channels=math.ceil(width_factor * 80/8) * 8,
              out_channels=math.ceil(width_factor * 112/8) * 8,
              expansion_factor=6,
              kernel_size=(5, 5),
              strides=(1, 1),
              name='Block_5')

    x = Block(x,
              depth=math.ceil(depth_factor * 4),
              in_channels=math.ceil(width_factor * 112/8) * 8,
              out_channels=math.ceil(width_factor * 192/8) * 8,
              expansion_factor=6,
              kernel_size=(5, 5),
              strides=(2, 2),
              name='Block_6')

    x = Block(x,
              depth=math.ceil(depth_factor * 1),
              in_channels=math.ceil(width_factor * 192/8) * 8,
              out_channels=math.ceil(width_factor * 320/8) * 8,
              expansion_factor=6,
              kernel_size=(3, 3),
              strides=(1, 1),
              name='Block_7')

    x = top(x,
            filters=math.ceil(width_factor * 1280/8) * 8)

    return x
