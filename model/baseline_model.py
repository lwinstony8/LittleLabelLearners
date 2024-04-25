import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# os.environ["KERAS_BACKEND"] = "tensorflow" # I don't think this is necessary


# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import keras
from keras import layers

from data.dataloader import Dataloader, download_data
from custom_callbacks import ScheduledSubsetCallback
import hyperparameters as hp

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
    def __init__(self, train, test):
        super().__init__()
        self.base_num_classes = 7
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
    my_baseline_model = BaselineModel(train, test)
    my_baseline_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")], 
    )
    baseline_logs = my_baseline_model.fit(
        my_baseline_model.dataloader.labeled_train_dataset, 
        epochs=hp.num_epochs, 
        validation_data=my_baseline_model.dataloader.test_dataset,
        callbacks=[ScheduledSubsetCallback()]
    )

    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(baseline_logs.history["val_acc"]) * 100
        )
    )