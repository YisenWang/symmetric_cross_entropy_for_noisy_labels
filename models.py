import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from resnet import cifar10_resnet
from keras.applications.resnet50 import ResNet50

def get_model(dataset='mnist', input_tensor=None, input_shape=None, num_classes=10):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar-10' or 'cifar-100') and returns the appropriate Keras model.
    :param dataset: A string indicating which dataset we are building
                    a model for.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    input_shape: optional shape tuple
    :return: The model; a Keras 'Model' instance.
    """
    assert dataset in ['mnist', 'cifar-10', 'cifar-100'], \
        "dataset parameter must be either 'mnist', 'cifar-10' or 'cifar-100'"

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_shape):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if dataset == 'mnist':
        # ## 
        x = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", name='conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        x = Flatten()(x)

        x = Dense(128, kernel_initializer="he_normal", name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)
        # x = Dropout(0.2)(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation(tf.nn.softmax)(x)

        model = Model(img_input, x)

    elif dataset == 'cifar-10':
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Flatten(name='flatten')(x)

        x = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation(tf.nn.softmax)(x)

        # Create model.
        model = Model(img_input, x)


    elif dataset == 'cifar-100':
        model = cifar10_resnet(depth=7, num_classes=num_classes)

    return model