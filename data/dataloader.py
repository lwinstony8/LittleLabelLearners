import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Dataset hyperparameters
unlabeled_dataset_size = 100000
labeled_dataset_size = 5000
image_channels = 3

# Algorithm hyperparameters
num_epochs = 20
batch_size = 525  # Corresponds to 200 steps per epoch
width = 128
temperature = 0.1
# dataloader class used to load in all our data
class Dataloader():

    # constructor for the Dataloader
    def __init__(self, train, test):
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test

        # num_classes refers to the ceiling amount of distinct classes to learn
        # theoretically, should be very high; for testing purposes, we will set it to 10
        # since we know the world (i.e. CIFAR10 dataset) only has 10 objects
        self.num_classes = len(np.unique(self.y_test))

        self.train_dataset = None
        self.labeled_train_dataset = None
        self.test_dataset = None
        
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
        self.y_test_subset = self.y_test[subset_labels_test_indices]
    
    # return the subsets that we just generated

    # TODO: we only ever use this in visualizations.py; is this a problem??
    def get_subsets(self):
        return tf.cast(self.x_train_subset, dtype=tf.float32), tf.cast(self.y_train_subset, dtype=tf.float32), tf.cast(self.x_test_subset, dtype=tf.float32), tf.cast(self.y_test_subset, dtype=tf.float32)
    
    def generate_labeled_unlabeled_indices(self, data, split_rate=0.5):
        # getting the number of samples
        num_samples = len(data)
        rng = np.random.default_rng()
        # randomly choosing indices that are the percentage of the splitrate
        labeled_indices = rng.choice(
            np.arange(num_samples), 
            int(num_samples * split_rate), 
            replace=False)
        
        # getting the other indices that are not a part of the labeled
        unlabeled_indices = np.nonzero(
            np.isin(np.arange(num_samples), labeled_indices, invert=True))[0]
        
        # print(f'{num_samples=}')
        # print(f'{labeled_indices.shape=}')
        # print(f'{unlabeled_indices.shape=}')

        # returning the indices for labeld and unlabeled
        return labeled_indices, unlabeled_indices 

        
    # preprocessing the data, normalizing all the values
    def preprocess(self):
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.

    def prepare_dataset(self, x_train, y_train, x_test, y_test):
        # Labeled and unlabeled samples are loaded synchronously
        # with batch sizes selected accordingly
        steps_per_epoch = (unlabeled_dataset_size + labeled_dataset_size) // batch_size
        unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
        labeled_batch_size = labeled_dataset_size // steps_per_epoch
        print(
            f"batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)"
        )
        # getting the indices for out labeled and unlabeled data
        train_x_labeled_idx, train_x_unlabeled_idx = self.generate_labeled_unlabeled_indices(x_train)

        # getting the unlable
        unlabeled_train_dataset = x_train[train_x_unlabeled_idx]
        unlabeled_train_dataset = (
            tf.data.Dataset.from_tensor_slices(unlabeled_train_dataset)\
            # .shuffle()
            .batch(unlabeled_batch_size))
        
        #
        train_x_labeled = x_train[train_x_labeled_idx]
        train_y_labeled = y_train[train_x_labeled_idx]
        
        # labeled_train_dataset = (
        #     tf.data.Dataset.from_tensor_slices((train_x_labeled, self.one_hot(train_y_labeled)))
        #     # .shuffle()
        #     .batch(labeled_batch_size)
        # )

        # test_dataset = (
        #     tf.data.Dataset.from_tensor_slices((x_test, self.one_hot(y_test)))
        #     # .shuffle()
        #     .batch(batch_size)
        # )

        labeled_train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_x_labeled, train_y_labeled))
            # .shuffle()
            .batch(labeled_batch_size)
        )

        test_dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            # .shuffle()
            .batch(batch_size)
        )

        # Labeled and unlabeled datasets are zipped together
        train_dataset = tf.data.Dataset.zip(
            (unlabeled_train_dataset, labeled_train_dataset)
        )
        # prefetch if we need to
        # train_dataset in tuple of itself to match
        self.train_dataset=train_dataset
        self.labeled_train_dataset=labeled_train_dataset
        self.test_dataset=test_dataset
        # return train_dataset, labeled_train_dataset, test_dataset
    
    '''
    # method that one_hot encodes the labels for a non specific 
    def one_hot(self, labels):
        #print(f"{labels.shape=}")
        encoded = tf.one_hot(labels, depth=self.num_classes)
        encoded = tf.squeeze(encoded)
        print(f'{encoded.shape=}')
        return encoded
    '''

    

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
    dataloader.generate_labeled_unlabeled_indices(x_train_subset, 0.321)
    #print(f'{x_train_subset.shape=}')

    

    