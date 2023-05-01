import tensorflow as tf

from keras.callbacks import Callback


class TrainCallback(Callback):
    def __init__(self, model, batches, epochs, accuracy_threshold=0.9, target=0.95, lr_factor=0.1, reset=True, epoch_patience=1, lr_patience=1):
        super(TrainCallback, self).__init__()
        self.model = model
        self.best_weights = self.model.get_weights()

        self.batches = batches
        self.epochs = epochs
        self.accuracy_threshold = accuracy_threshold
        self.target = target
        self.lr_factor = lr_factor
        self.reset = reset
        self.epoch_patience = epoch_patience
        self.lr_patience = lr_patience

        self.bad_epoch_count = 0
        self.bad_lr_count = 0
        self.best_epoch = 0
        self.best_acc = 0
        self.best_val_acc = 0

        self.logs = []

    def good_epoch_update(self, epoch):
        self.best_weights = self.model.get_weights()
        self.best_epoch = epoch

        self.bad_epoch_count = 0
        self.bad_lr_count = 0
        self.logs.append(f'{epoch}: Good Epoch')

    def bad_epoch_update(self, epoch):
        self.bad_epoch_count += 1
        if self.bad_epoch_count == self.epoch_patience:
            self.bad_epoch_count = 0
            self.bad_lr_count += 1
            if self.bad_lr_count == self.lr_patience:
                self.logs.append(f'{epoch}: Training stopped at epoch {epoch+1} due to no further improvement')
                self.model.stop_training = True
            else:
                self.logs.append(f'{epoch}: Adjusting Learning Rate')
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr*self.lr_factor)

            if self.reset:
                self.model.set_weights(self.best_weights)
                self.logs.append(f'{epoch}: Resetting weights to epoch {self.best_epoch+1} weights')
        self.logs.append(f'{epoch}: Bad Epoch {self.bad_epoch_count}')
        self.logs.append(f'{epoch}: Bad learning rate {self.bad_lr_count}')

    def on_epoch_end(self, epoch, logs=None):
        acc = round(logs['accuracy'], 2)
        val_acc = round(logs['val_accuracy'], 2)

        if val_acc >= self.target:
            self.best_weights = self.model.get_weights()
            self.model.stop_training = True
            self.logs.append(f'{epoch}: Target Accuracy Achieved')

        else:
            if acc < self.accuracy_threshold:
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.good_epoch_update(epoch)
                else:
                    self.bad_epoch_update(epoch)

                self.best_val_acc = max(self.best_val_acc, val_acc)

            else:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.good_epoch_update(epoch)
                else:
                    self.bad_epoch_update(epoch)

                self.best_acc = max(self.best_acc, acc)
