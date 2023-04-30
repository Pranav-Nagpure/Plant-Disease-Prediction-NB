import os
import math
import matplotlib
import numpy as np
import tensorflow as tf
import ipywidgets as widgets
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from ipywidgets.embed import embed_minimal_html
from IPython.display import display, clear_output, HTML


class PrintCallback(Callback):
    def __init__(self, model, batches, epochs, accuracy_threshold=90, target=95, lr_factor=0.1, reset=True, epoch_patience=1, lr_patience=1):
        super(PrintCallback, self).__init__()
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

    def on_train_begin(self, logs=None):
        print('Initializing Training')

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        print('Training Complete')

    def on_train_batch_end(self, batch, logs=None):
        acc = logs['accuracy'] * 100
        loss = logs['loss']
        print(f'Processing Batch {batch} of {self.batches:.0f}, accuracy = {acc:6.2f}%, loss = {loss:5.2f}'.ljust(100), '\r', end='')

    def good_epoch_update(self, epoch):
        self.best_weights = self.model.get_weights()
        self.best_epoch = epoch

        self.bad_epoch_count = 0
        self.bad_lr_count = 0

    def bad_epoch_update(self, epoch, messages):
        self.bad_epoch_count += 1
        if self.bad_epoch_count == self.epoch_patience:
            self.bad_epoch_count = 0
            self.bad_lr_count += 1
            if self.bad_lr_count == self.lr_patience:
                messages.append(f'Training stopped at epoch {epoch+1} due to no further improvement')
                self.model.stop_training = True
            else:
                messages.append(f'Adjusting Learning Rate')
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr*self.lr_factor)

            if self.reset:
                self.model.set_weights(self.best_weights)
                messages.append(f'Resetting weights to epoch {self.best_epoch+1} weights')

    def on_epoch_end(self, epoch, logs=None):
        messages = []

        acc = round(logs['accuracy'] * 100, 2)
        loss = logs['loss']
        val_acc = round(logs['val_accuracy'] * 100, 2)
        val_loss = logs['val_loss']

        if val_acc >= self.target:
            self.model.stop_training = True
            messages.append('Target Accuracy Achieved')

        else:
            if acc < self.accuracy_threshold:
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.good_epoch_update(epoch)
                else:
                    self.bad_epoch_update(epoch, messages)

                self.best_val_acc = max(self.best_val_acc, val_acc)

            else:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.good_epoch_update(epoch)
                else:
                    self.bad_epoch_update(epoch, messages)

                self.best_acc = max(self.best_acc, acc)

        print(f'Epoch: {epoch+1}/{self.epochs}, acc = {acc:6.2f}%, loss = {loss:5.2f}, val_acc = {val_acc:6.2f}%, val_loss = {val_loss:5.2f}'.ljust(100))
        for message in messages:
            print(message)


class ProgressDisplay():
    def __init__(self, batches, val_batches, epochs):
        self.batches = batches
        self.val_batches = val_batches
        self.epochs = epochs

        self.epoch_progress = widgets.IntProgress(value=0,
                                                  min=0,
                                                  max=self.batches,
                                                  bar_style='info',
                                                  style={'bar_color': 'blue'},
                                                  orientation='horizontal')

        self.epoch_label = widgets.Label(value=f'Initializing Training')

        self.val_progress = widgets.IntProgress(value=0,
                                                min=0,
                                                max=self.val_batches,
                                                bar_style='info',
                                                style={'bar_color': 'blue'},
                                                orientation='horizontal')

        self.val_label = widgets.Label(value=f'Initializing Training')

        self.training_progress = widgets.IntProgress(value=0,
                                                     min=0,
                                                     max=self.epochs,
                                                     bar_style='info',
                                                     style={'bar_color': 'blue'},
                                                     orientation='horizontal')

        self.training_label = widgets.Label(value=f'Epoch 0 of {self.epochs}')

        self.accuracy_bar = widgets.FloatProgress(value=0.0,
                                                  min=0.0,
                                                  max=1.0,
                                                  bar_style='info',
                                                  style={'bar_color': 'green'},
                                                  orientation='vertical')

        self.accuracy_label = widgets.Label(value=f'{0.0:.2f}%')

        self.val_accuracy_bar = widgets.FloatProgress(value=0,
                                                      min=0.0,
                                                      max=1.0,
                                                      bar_style='info',
                                                      style={'bar_color': 'lime'},
                                                      orientation='vertical')

        self.val_accuracy_label = widgets.Label(value=f'{0.0:.2f}%')

        self.epoch_block = widgets.VBox([widgets.HBox([widgets.Label(value='Epoch Progress: '),
                                                       widgets.VBox([self.epoch_progress,
                                                                     self.epoch_label],
                                                                    layout=widgets.Layout(align_items='center'))])])

        self.val_block = widgets.VBox([widgets.HBox([widgets.Label(value='Validation Progress: '),
                                                    widgets.VBox([self.val_progress,
                                                                  self.val_label],
                                                                 layout=widgets.Layout(align_items='center'))])])

        self.training_block = widgets.VBox([widgets.HBox([widgets.Label(value='Training Progress: '),
                                                          widgets.VBox([self.training_progress,
                                                                        self.training_label],
                                                                       layout=widgets.Layout(align_items='center'))])])

        self.accuracy_block = widgets.VBox([widgets.Label(value='Training'),
                                            widgets.Label(value='Accuracy'),
                                            self.accuracy_bar,
                                            self.accuracy_label],
                                           layout=widgets.Layout(align_items='center'))

        self.val_accuracy_block = widgets.VBox([widgets.Label(value='Validation'),
                                                widgets.Label(value='Accuracy'),
                                                self.val_accuracy_bar,
                                                self.val_accuracy_label],
                                               layout=widgets.Layout(align_items='center'))

        self.image_block = widgets.Image()
        self.update_curve([], [])

        self.display_block = widgets.HBox([widgets.VBox([self.epoch_block,
                                                         self.val_block,
                                                         self.training_block],
                                                        layout=widgets.Layout(display='inline-flex',
                                                                              align_items='flex-end')),
                                           self.accuracy_block,
                                           self.val_accuracy_block,
                                           self.image_block],
                                          layout=widgets.Layout(align_items='center'))

        display(self.display_block)

    def increase_batch(self):
        self.epoch_progress.value += 1

    def set_batch_label(self, batch):
        self.epoch_label.value = f'Batch {batch} of {self.batches}'

    def increase_val_batch(self):
        self.val_progress.value += 1

    def set_val_batch_label(self, val_batch):
        self.val_label.value = f'Batch {val_batch} of {self.val_batches}'

    def increase_epoch(self):
        self.training_progress.value += 1
        self.epoch_progress.value = 0
        self.val_progress.value = 0

    def set_epoch_label(self, epoch):
        self.training_label.value = f'Epoch {epoch} of {self.epochs}'

    def set_training(self):
        self.val_label.value = 'Training'

    def set_validation(self):
        self.epoch_label.value = 'Validating'

    def update_accuracy(self, acc):
        self.accuracy_bar.value = acc
        self.accuracy_label.value = f'{acc*100:.2f}%'

        G = int(acc * 255)
        R = 255 - G
        B = 0
        self.accuracy_bar.style = {'bar_color': f'#{R:02x}{G:02x}{B:02x}'}

    def update_val_accuracy(self, val_acc):
        self.val_accuracy_bar.value = val_acc
        self.val_accuracy_label.value = f'{val_acc*100:.2f}%'

        G = int(val_acc * 255)
        R = 255 - G
        B = 0
        self.val_accuracy_bar.style = {'bar_color': f'#{R:02x}{G:02x}{B:02x}'}

    def set_training_complete(self, early):
        if early:
            self.training_label.value = f'Early Stop Training on epoch {self.training_progress.value}'
        else:
            self.training_label.value = 'Training Complete'

        self.epoch_label.value = 'Training Complete'
        self.val_label.value = 'Training Complete'

    def update_curve(self, accuracies, val_accuracies, best_epoch=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title('Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_xlim(0, math.ceil((self.training_progress.value+1)/10)*10)
        ax.set_ylim(0, 1)
        ax.plot(np.arange(1, len(accuracies)+1), accuracies, label='train')
        ax.plot(np.arange(1, len(val_accuracies)+1), val_accuracies, label='val')
        if best_epoch != None:
            ax.plot([best_epoch+1, best_epoch+1], [accuracies[best_epoch], val_accuracies[best_epoch]], 'go', label='Best Epoch')
        ax.legend()
        fig.savefig('curve.png')
        plt.close(fig)
        file = open('curve.png', 'rb')
        self.image_block.value = file.read()
        file.close()
        os.remove('curve.png')


class DisplayCallback(Callback):
    def __init__(self, model, batches, val_batches, epochs, accuracy_threshold=90, target=95, lr_factor=0.1, reset=True, epoch_patience=1, lr_patience=1):
        super(DisplayCallback, self).__init__()
        self.model = model
        self.best_weights = self.model.get_weights()

        self.batches = batches
        self.val_batches = val_batches
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
        self.early_stop = False

        self.accuracies = []
        self.val_accuracies = []

        self.progress_display = None

    def on_train_begin(self, logs=None):
        matplotlib.use('Agg')
        self.progress_display = ProgressDisplay(self.batches, self.val_batches, self.epochs)

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

        self.progress_display.set_training_complete(self.early_stop)

        # For statically rendering widgets
        embed_minimal_html('result_display.html', views=[self.progress_display.display_block])
        clear_output()
        display(HTML('result_display.html'))
        os.remove('result_display.html')

        matplotlib.use('module://matplotlib_inline.backend_inline')

    def on_train_batch_begin(self, batch, logs=None):
        self.progress_display.set_batch_label(batch+1)

    def on_train_batch_end(self, batch, logs=None):
        self.progress_display.increase_batch()
        self.progress_display.update_accuracy(logs['accuracy'])

    def good_epoch_update(self, epoch):
        self.best_weights = self.model.get_weights()
        self.best_epoch = epoch

        self.bad_epoch_count = 0
        self.bad_lr_count = 0

    def bad_epoch_update(self):
        self.bad_epoch_count += 1
        if self.bad_epoch_count == self.epoch_patience:
            self.bad_epoch_count = 0
            self.bad_lr_count += 1
            if self.bad_lr_count == self.lr_patience:
                self.model.stop_training = True
                self.early_stop = True
            else:
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr*self.lr_factor)

            if self.reset:
                self.model.set_weights(self.best_weights)

    def on_test_batch_begin(self, batch, logs=None):
        self.progress_display.set_val_batch_label(batch+1)

    def on_test_batch_end(self, batch, logs=None):
        self.progress_display.increase_val_batch()
        self.progress_display.update_val_accuracy(logs['accuracy'])

    def on_test_begin(self, logs=None):
        self.progress_display.set_validation()

    def on_epoch_begin(self, epoch, logs=None):
        self.progress_display.set_epoch_label(epoch+1)
        self.progress_display.set_training()

    def on_epoch_end(self, epoch, logs=None):
        acc = round(logs['accuracy'] * 100, 2)
        val_acc = round(logs['val_accuracy'] * 100, 2)

        if val_acc >= self.target:
            self.model.stop_training = True

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
                    self.bad_epoch_update()

                self.best_acc = max(self.best_acc, acc)

        self.progress_display.increase_epoch()
        self.accuracies.append(logs['accuracy'])
        self.val_accuracies.append(logs['val_accuracy'])
        self.progress_display.update_curve(self.accuracies, self.val_accuracies, self.best_epoch)
