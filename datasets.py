import os
import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import np_utils
from util import other_class
from numpy.testing import assert_array_almost_equal

# Set random seed
np.random.seed(123)

NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100}

def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = np.eye(size)
    cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
    P[cls1, cls2] = noise
    P[cls2, cls1] = noise
    P[cls1, cls1] = 1.0 - noise
    P[cls2, cls2] = 1.0 - noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def get_data(dataset='mnist', noise_ratio=0, asym=False, random_shuffle=False):
    """
    Get training images with specified ratio of syn/ayn label noise
    """
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

    elif dataset == 'cifar-10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    elif dataset == 'cifar-100':
        # num_classes = 100
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()

        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()
    else:
        return None, None, None, None


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train_clean = np.copy(y_train)
    # generate random noisy labels
    if noise_ratio > 0:
        if asym:
            data_file = "data/asym_%s_train_labels_%s.npy" % (dataset, noise_ratio)
            if dataset == 'cifar-100':
                P_file = "data/asym_%s_P_value_%s.npy" % (dataset, noise_ratio)
        else:
            data_file = "data/%s_train_labels_%s.npy" % (dataset, noise_ratio)
        if os.path.isfile(data_file):
            y_train = np.load(data_file)
            if dataset == 'cifar-100' and asym:
                P = np.load(P_file)
        else:
            if asym:
                if dataset == 'mnist':
                    # 1 < - 7, 2 -> 7, 3 -> 8, 5 <-> 6
                    source_class = [7, 2, 3, 5, 6]
                    target_class = [1, 7, 8, 6, 5]
                elif dataset == 'cifar-10':
                    # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
                    source_class = [9, 2, 3, 5, 4]
                    target_class = [1, 0, 5, 3, 7]

                elif dataset == 'cifar-100':
                        P = np.eye(NUM_CLASSES[dataset])
                        n = noise_ratio/100.0
                        nb_superclasses = 20
                        nb_subclasses = 5

                        if n > 0.0:
                            for i in np.arange(nb_superclasses):
                                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                                P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                            y_train_noisy = multiclass_noisify(y_train, P=P,
                                                               random_state=0)
                            actual_noise = (y_train_noisy != y_train).mean()
                            assert actual_noise > 0.0
                            y_train = y_train_noisy
                        np.save(P_file, P)

                else:
                    print('Asymmetric noise is not supported now for dataset: %s' % dataset)
                    return

                if dataset == 'mnist' or dataset == 'cifar-10':
                    for s, t in zip(source_class, target_class):
                        cls_idx = np.where(y_train_clean == s)[0]
                        n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)
                        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                        y_train[noisy_sample_index] = t

            else:
                n_samples = y_train.shape[0]
                n_noisy = int(noise_ratio * n_samples / 100)
                class_index = [np.where(y_train_clean == i)[0] for i in range(NUM_CLASSES[dataset])]
                class_noisy = int(n_noisy / NUM_CLASSES[dataset])

                noisy_idx = []
                for d in range(NUM_CLASSES[dataset]):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)

                for i in noisy_idx:
                    y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
            np.save(data_file, y_train)


        # print statistics
        print("Print noisy label generation statistics:")
        for i in range(NUM_CLASSES[dataset]):
            n_noisy = np.sum(y_train == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))

    if random_shuffle:
        # random shuffle
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train, y_train_clean = X_train[idx_perm], y_train[idx_perm], y_train_clean[idx_perm]

    # one-hot-encode the labels
    y_train_clean = np_utils.to_categorical(y_train_clean, NUM_CLASSES[dataset])
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test", y_test.shape)

    return X_train, y_train, y_train_clean, X_test, y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data(dataset='mnist', noise_ratio=40)
    Y_train = np.argmax(Y_train, axis=1)
    (_, Y_clean_train), (_, Y_clean_test) = mnist.load_data()
    clean_selected = np.argwhere(Y_train == Y_clean_train).reshape((-1,))
    noisy_selected = np.argwhere(Y_train != Y_clean_train).reshape((-1,))
    print("#correct labels: %s, #incorrect labels: %s" % (len(clean_selected), len(noisy_selected)))