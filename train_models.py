from __future__ import absolute_import
from __future__ import print_function

import os
import time
import numpy as np
import keras.backend as K
import argparse
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from util import get_lr_scheduler
from datasets import get_data
from models import get_model
from loss import symmetric_cross_entropy, cross_entropy, lsr, joint_optimization_loss, generalized_cross_entropy, boot_soft, boot_hard, forward, backward
from callback_util import LoggerCallback, SGDLearningRateTracker


# prepare folders
folders = ['data', 'model', 'log']
for folder in folders:
    path = os.path.join('./', folder)
    if not os.path.exists(path):
        os.makedirs(path)

def train(dataset='mnist', model_name='sl', batch_size=128, epochs=50, noise_ratio=0, asym=False, alpha = 1.0, beta = 1.0):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param dataset: 
    :param model_name:
    :param batch_size: 
    :param epochs: 
    :param noise_ratio: 
    :return: 
    """
    print('Dataset: %s, model: %s, batch: %s, epochs: %s, noise ratio: %s%%, asymmetric: %s, alpha: %s, beta: %s' %
          (dataset, model_name, batch_size, epochs, noise_ratio, asym, alpha, beta))

    # load data
    X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio, asym=asym, random_shuffle=False)
    n_images = X_train.shape[0]
    image_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    print("n_images", n_images, "num_classes", num_classes, "image_shape:", image_shape)

    # load model
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=num_classes)
    # model.summary()

    optimizer = SGD(lr=0.1, decay=5e-3, momentum=0.9)

    # create loss
    if model_name == 'ce':
        loss = cross_entropy
    elif model_name =='sl':
        loss = symmetric_cross_entropy(alpha,beta)
    elif model_name == 'lsr':
        loss = lsr
    elif model_name =='joint':
        loss = joint_optimization_loss
    elif model_name =='gce':
        loss = generalized_cross_entropy
    elif model_name == 'boot_hard':
        loss = boot_hard
    elif model_name == 'boot_soft':
        loss = boot_soft
    elif model_name == 'forward':
        loss = forward(P)
    elif model_name == 'backward':
        loss = backward(P)
    else:
        print("Model %s is unimplemented!" % model_name)
        exit(0)

    # model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    if asym:
        model_save_file = "model/asym_%s_%s_%s.{epoch:02d}.hdf5" % (model_name, dataset, noise_ratio)
    else:
        model_save_file = "model/%s_%s_%s.{epoch:02d}.hdf5" % (model_name, dataset, noise_ratio)


    ## do real-time updates using callbakcs
    callbacks = []

    if model_name == 'sl':
        cp_callback = ModelCheckpoint(model_save_file,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=False,
                                      save_weights_only=True,
                                      period=1)
        callbacks.append(cp_callback)
    else:
        cp_callback = ModelCheckpoint(model_save_file,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=False,
                                      save_weights_only=True,
                                      period=1)
        callbacks.append(cp_callback)

    # learning rate scheduler if use sgd
    lr_scheduler = get_lr_scheduler(dataset)
    callbacks.append(lr_scheduler)

    callbacks.append(SGDLearningRateTracker(model))

    # acc, loss, lid
    log_callback = LoggerCallback(model, X_train, y_train, y_train_clean, X_test, y_test, dataset, model_name, noise_ratio, asym, epochs, alpha, beta)
    callbacks.append(log_callback)

    # data augmentation
    if dataset in ['mnist', 'svhn']:
        datagen = ImageDataGenerator()
    elif dataset in ['cifar-10']:
        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    datagen.fit(X_train)

    # train model
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=callbacks
                        )

def main(args):
    train(args.dataset, args.model_name, args.batch_size, args.epochs, args.noise_ratio, args.asym, args.alpha, args.beta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar-10', 'cifar-100'",
        required=True, type=str
    )
    parser.add_argument(
        '-m', '--model_name',
        help="Model name: 'ce', 'sl' ",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-r', '--noise_ratio',
        help="The percentage of noisy labels [0, 100].",
        required=False, type=int
    )
    parser.add_argument(
        '-a', '--asym',
        help="asymmetric noise.",
        required=False, type=bool
    )
    parser.add_argument(
        '-alpha', '--alpha',
        help="alpha parameter.",
        required=True, type=float
    )
    parser.add_argument(
        '-beta', '--beta',
        help="beta parameter.",
        required=True, type=float
    )
    parser.set_defaults(epochs=150)
    parser.set_defaults(batch_size=128)
    parser.set_defaults(noise_ratio=0)
    parser.set_defaults(asym=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    # MNIST

    # args = parser.parse_args(['-d', 'mnist', '-m', 'sl',
    #                           '-e', '50', '-b', '128',
    #                           '-r', '40', '-alpha', '0.01', '-beta', '1.0'])
    # main(args)


    # CIFAR-10

    args = parser.parse_args(['-d', 'cifar-10', '-m', 'sl',
                              '-e', '120', '-b', '128',
                              '-r', '40', '-alpha', '0.1', '-beta', '1.0'])
    main(args)

    # CIFAR-100

    # args = parser.parse_args(['-d', 'cifar-100', '-m', 'sl',
    #                           '-e', '150', '-b', '128',
    #                           '-r', '40', '-alpha', '6.0', '-beta', '0.1'])
    # main(args)

    K.clear_session()
