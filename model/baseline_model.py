import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# os.environ["KERAS_BACKEND"] = "tensorflow" # I don't think this is necessary


# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import numpy as np
import keras
from keras import layers

from data.dataloader import Dataloader, download_data
from custom_callbacks import ScheduledSubsetCallback
import hyperparameters as hp

from collections import defaultdict

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

class BaselineModel(keras.Model):
    def __init__(self, train: np.ndarray, test: np.ndarray, 
                 num_classes_range: int | tuple[int, int]=(5, 10), 
                 split_rate_range: float | tuple[float, float]=(0.5, 0.5),
                 contrastive_learning_rate_range: float | tuple[float, float]=(0.001,0.001),
                 probe_learning_rate_range: float | tuple[float, float]=(0.01,0.01)):
        """ Initializer for BaselineModel. Simple CNN with linear classification head

        Args:
            train (np.ndarray): Training dataset containing labeled and unlabeled samples
            test (np.ndarray): Testing dataset containing labeled and unlabeled samples
            num_classes_range (int | tuple[int, int], optional): INCLUSIVE range of possible subset_sizes. Use a single int to designate constant subset_size. Defaults to (5,10).
            split_rate_range (float | tuple[float, float], optional): INCLUSIVE range of possible split_rates. Use a single float to designate constant split_rate. Defaults to (0.5, 0.5).
        """        
        super().__init__()
        self.num_classes_range = num_classes_range if isinstance(num_classes_range, tuple) else (num_classes_range, num_classes_range)
        self.split_rate_range= split_rate_range if isinstance(split_rate_range, tuple) else (split_rate_range, split_rate_range)
        self.cur_num_classes = self.num_classes_range[0]
        self.cur_split_rate = self.split_rate_range[0]

        # need to include these so that the baseline can run after doing all the scheduler changes for gradual
        self.contrastive_learning_rate_range = contrastive_learning_rate_range if isinstance(contrastive_learning_rate_range, tuple) else (contrastive_learning_rate_range, contrastive_learning_rate_range)
        self.curr_contrastive_learning_rate = self.contrastive_learning_rate_range[1]
        self.probe_learning_rate_range = probe_learning_rate_range if isinstance(probe_learning_rate_range, tuple) else (probe_learning_rate_range, probe_learning_rate_range)
        self.curr_probe_learning_rate = self.probe_learning_rate_range[1]
        # self.floor_num_classes = floor_num_classes
        # self.ceiling_num_classes = ceiling_num_classes
        self.dataloader = Dataloader(train,test)
        self.dataloader.preprocess()
        self.dataloader.generate_subsets(self.cur_num_classes)
        self.dataloader.prepare_dataset(
            self.dataloader.x_train_subset, 
            self.dataloader.y_train_subset, 
            self.dataloader.x_test, 
            self.dataloader.y_test,
            split_rate=self.cur_split_rate)
        
        
        
        #define the layers for the model
        self.encoder = keras.Sequential(
        [
            layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(hp.width, activation="relu"),
        ],
        name="encoder",)
        self.dense = layers.Dense(self.dataloader.num_classes)
        
    def call(self, inputs):
        inputs = self.encoder(inputs)
        inputs = self.dense(inputs)
        return inputs
        
if __name__ == '__main__':
    # TODO: parseargs; check if we want scheduled subset; set base_num_classes...
    train, test = download_data()

    my_baseline_model = BaselineModel(train, test, 
                                      num_classes_range=(5, 10),
                                      split_rate_range=(0.5, 0.5))
    my_baseline_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")], 
    )

    # TODO: Write our own train/test to take in indices instead of dataset objects
    # Therefore, we can use the same original data object, but only calculate 
    # indices to access existing objects

    model_history = defaultdict(lambda: [])
    scheduled_subset_callback = ScheduledSubsetCallback(my_baseline_model)
    for epoch in range(hp.num_epochs):
        scheduled_subset_callback(cur_epoch=epoch)
        for k, v in my_baseline_model.fit(
            my_baseline_model.dataloader.labeled_train_dataset, 
            epochs=1, 
            validation_data=my_baseline_model.dataloader.test_dataset,
        ).history.items():
            model_history[k].extend(v)

    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(model_history["val_acc"]) * 100
        )
    )