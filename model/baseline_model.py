import os

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

os.environ["KERAS_BACKEND"] = "tensorflow"


# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import math
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_datasets as tfds

import keras
from keras import ops
from keras import layers

from augmentations import RandomColorAffine, get_augmenter
from data.dataloader import Dataloader, download_data

# Stronger augmentations for contrastive, weaker ones for supervised training
contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {
    "min_area": 0.75,
    "brightness": 0.3,
    "jitter": 0.1,
}
# Algorithm hyperparameters
num_epochs = 20
batch_size = 525  # Corresponds to 200 steps per epoch
width = 128
temperature = 0.1

train, test = download_data()
# my_dataloader = Dataloader(train, test)

# my_dataloader.preprocess()
# my_dataloader.generate_subsets()

# y_train_onehot, y_test_onehot = my_dataloader.one_hot(my_dataloader.y_train, my_dataloader.y_test)


'''
# Full Train/Test data
train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
    my_dataloader.x_train, 
    my_dataloader.y_train, 
    my_dataloader.x_test, 
    my_dataloader.y_test)
'''

'''
# Subset Train; Full Test data
train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
    my_dataloader.x_train_subset, 
    my_dataloader.y_train_subset, 
    my_dataloader.x_test, 
    my_dataloader.y_test)
'''

'''
# Subset Train/Test data
train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
    my_dataloader.x_train_subset, 
    my_dataloader.y_train_subset, 
    my_dataloader.x_test_subset, 
    my_dataloader.y_test_subset)
'''

'''
# Full Train; Subset Test data
train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
    my_dataloader.x_train, 
    my_dataloader.y_train, 
    my_dataloader.x_test_subset, 
    my_dataloader.y_test_subset)
'''

# my_dataloader.prepare_dataset(
#     my_dataloader.x_train, 
#     my_dataloader.y_train, 
#     my_dataloader.x_test_subset, 
#     my_dataloader.y_test_subset)

# # Define the encoder architecture
# def get_encoder():
#     return keras.Sequential(
#         [
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
#             layers.Flatten(),
#             layers.Dense(width, activation="relu"),
#         ],
#         name="encoder",
#     )

# # Baseline supervised training with random initialization
# baseline_model = keras.Sequential(
#     [
#         # get_augmenter(**classification_augmentation),
#         get_encoder(),
#         layers.Dense(10),
#     ],
#     name="baseline_model",
# )
# baseline_model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
# )

# baseline_history = baseline_model.fit(
#     my_dataloader.labeled_train_dataset, epochs=num_epochs, validation_data=my_dataloader.test_dataset
# )

class BaselineModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.base_num_classes = 7
        self.train, self.test = download_data()
        self.dataloader = Dataloader(train,test)
        self.dataloader.preprocess()
        self.dataloader.generate_subsets(self.base_num_classes)
        self.dataloader.prepare_dataset(
            self.dataloader.x_train_subset, 
            self.dataloader.y_train_subset, 
            self.dataloader.x_test, 
            self.dataloader.y_test)
        
        #define the layers for the model
        self.encoder = keras.Sequential(
        [
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",)
        self.dense = layers.Dense(self.dataloader.num_classes)
        
    def call(self, inputs):
        inputs = self.encoder(inputs)
        inputs = self.dense(inputs)
        return inputs



class ScheduledSubsetCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        print('reached callback')
        
    def on_epoch_begin(self, epoch, logs=None):
        subset_size = min(epoch, self.model.dataloader.num_classes)
        # print(f'{type(subset_size)=}')
        # print(f'{type(epoch)=}')
        # print(f'{type(self.model.base_num_classes)=}')
        subset_size = max(subset_size, self.model.base_num_classes) # begin with 5 classes
        print(f'{subset_size=}')
        self.model.dataloader.generate_subsets(subset_size=subset_size)
        ## NOTE: this might be put inside ScheduledSubsetCallback
        # this will get replaced once we have the other scheduler
        self.model.dataloader.prepare_dataset(
            self.model.dataloader.x_train_subset, 
            self.model.dataloader.y_train_subset, 
            self.model.dataloader.x_test, 
            self.model.dataloader.y_test)
        print(f'{self.model.dataloader.x_train_subset.shape=}')
        
my_baseline_model = BaselineModel()
my_baseline_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")], 
)
baseline_logs = my_baseline_model.fit(
    my_baseline_model.dataloader.labeled_train_dataset, 
    epochs=20, 
    validation_data=my_baseline_model.dataloader.test_dataset,
    callbacks=[ScheduledSubsetCallback()]
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(baseline_logs.history["val_acc"]) * 100
    )
)