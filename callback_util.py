import numpy as np
import keras.backend as K
from keras.callbacks import Callback, LearningRateScheduler

class LoggerCallback(Callback):
    """
    Log train/val loss and acc into file for later plots.
    """
    def __init__(self, model, X_train, y_train, y_train_clean, X_test, y_test, dataset,
                 model_name, noise_ratio, asym, epochs, alpha, beta):
        super(LoggerCallback, self).__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_clean = y_train_clean
        self.X_test = X_test
        self.y_test = y_test
        self.n_class = y_train.shape[1]
        self.dataset = dataset
        self.model_name = model_name
        self.noise_ratio = noise_ratio
        self.asym = asym
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.train_loss_class = [None]*self.n_class
        self.train_acc_class = [None]*self.n_class

        # the followings are used to estimate LID
        self.lid_k = 20
        self.lid_subset = 128
        self.lids = []

        # complexity - Critical Sample Ratio (csr)
        self.csr_subset = 500
        self.csr_batchsize = 100
        self.csrs = []

    def on_epoch_end(self, epoch, logs={}):
        tr_acc = logs.get('acc')
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')

        self.train_loss.append(tr_loss)
        self.test_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(val_acc)

        print('ALL acc:', self.test_acc)

        if self.asym:
            file_name = 'log/asym_loss_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
            file_name = 'log/asym_acc_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))
            file_name = 'log/asym_class_loss_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.train_loss_class))
            file_name = 'log/asym_class_acc_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.train_acc_class))
        else:
            file_name = 'log/loss_%s_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio, self.alpha)
            np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
            file_name = 'log/acc_%s_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio, self.alpha)
            np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))

        return

class SGDLearningRateTracker(Callback):
    def __init__(self, model):
        super(SGDLearningRateTracker, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        init_lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        iterations = float(K.get_value(self.model.optimizer.iterations))
        lr = init_lr * (1. / (1. + decay * iterations))
        print('init lr: %.4f, current lr: %.4f, decay: %.4f, iterations: %s' % (init_lr, lr, decay, iterations))
