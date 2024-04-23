import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# dataloader class used to load in all our data
class Dataloader():

    # constructor for the Dataloader
    def __init__(self, train, test):
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test
        #print(f'{self.x_train.shape=}')

    # method to get the subsets of the labels
    def generate_subsets(self, subset_size=7):
        # creating the range of the labels that we want to select for
        subset_labels = np.arange(subset_size)
        subset_labels_train_indices = np.nonzero(np.isin(self.y_train, subset_labels))[0]
        subset_labels_test_indices = np.nonzero(np.isin(self.y_test, subset_labels))[0]

        # actually winindg down the data
        self.x_train_subset = self.x_train[subset_labels_train_indices]
        self.y_train_subset = self.y_train[subset_labels_train_indices]
        self.x_test_subset = self.x_test[subset_labels_test_indices]
        self.y_test_subset = self.x_test[subset_labels_test_indices]
    
    # return the subsets that we just generated
    def get_subsets(self):
        return tf.cast(self.x_train_subset, dtype=tf.float32), tf.cast(self.y_train_subset, dtype=tf.float32), tf.cast(self.x_test_subset, dtype=tf.float32), tf.cast(self.y_test_subset, dtype=tf.float32)

    # preprocessing the data, normalizing all the values
    def preprocess(self):
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.

    

# function that downloads the data from keras
def download_data():
    cifar10_dataset = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10_dataset
    
    return (x_train, y_train), (x_test, y_test)

# main method when the file is called
if __name__ == '__main__':
    train, test = download_data()
    dataloader = Dataloader(train, test)
    dataloader.preprocess()
    dataloader.generate_subsets()
    x_train_subset, y_train_subset, x_test_subset, y_test_subset = dataloader.get_subsets()
    #print(f'{x_train_subset.shape=}')

    

    