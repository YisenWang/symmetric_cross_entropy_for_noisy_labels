from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.callbacks import LearningRateScheduler

# Set random seed
np.random.seed(123)

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

def get_lr_scheduler(dataset):
    """
    customerized learning rate decay for training with clean labels.
     For efficientcy purpose we use large lr for noisy data.
    :param dataset: 
    :param noise_ratio:
    :return: 
    """
    if dataset in ['mnist']:
        def scheduler(epoch):
            if epoch > 30:
                return 0.001
            elif epoch > 10:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 80:
                return 0.0001
            elif epoch > 40:
                return 0.001
            else:
                return 0.01
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 120:
                return 0.001
            elif epoch > 80:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)

