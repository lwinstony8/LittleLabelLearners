import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Dataset hyperparameters
# unlabeled_dataset_size = 100000
# labeled_dataset_size = 5000
image_channels = 3

# Algorithm hyperparameters
num_epochs = 20
batch_size = 525  # Corresponds to 200 steps per epoch
width = 128
temperature = 0.1
# dataloader class used to load in all our data
class Dataloader():
    """ Dataloader Object contains the train/test data and handles data-specific operations, namely preprocessing, subsetting, defining label split rates, etc.

    """    

    # constructor for the Dataloader
    def __init__(self, train:np.ndarray, test:np.ndarray):
        """ Initializer for Dataloader

        Args:
            train (np.ndarray): Training dataset containing labeled and unlabeled samples
            test (np.ndarray): Testing dataset containing labeled and unlabeled samples
        """        
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test

        # num_classes refers to the ceiling amount of distinct classes to learn
        # theoretically, should be very high; for testing purposes, we will set it to 10
        # since we know the world (i.e. CIFAR10 dataset) only has 10 objects
        self.num_classes = len(np.unique(self.y_test))


        self.x_train_subset = None
        self.y_train_subset = None
        self.x_test_subset = None 
        self.y_test_subset = None 
        
        self.train_dataset = None
        self.labeled_train_dataset = None
        self.test_dataset = None
        #print(f'{self.x_train.shape=}')

    # method to get the subsets of the labels
    def generate_subsets(self, subset_size=7):
        """ Generates and save as instance variables train/test subsets for both features and labels. Subsets differ from the original data
            by how many num_classes are represented in the subsets

        Args:
            subset_size (int, optional): Defines the number of classes to keep in subset. Defaults to 7.
        """        
        
        
        # creating the range of the labels that we want to select for

        subset_labels = np.arange(subset_size)
        subset_labels_train_indices = np.nonzero(np.isin(self.y_train, subset_labels))[0]
        subset_labels_test_indices = np.nonzero(np.isin(self.y_test, subset_labels))[0]

        # actually winding down the data
        self.x_train_subset = self.x_train[subset_labels_train_indices]
        self.y_train_subset = self.y_train[subset_labels_train_indices]
        self.x_test_subset = self.x_test[subset_labels_test_indices]
        self.y_test_subset = self.y_test[subset_labels_test_indices]
    
    # return the subsets that we just generated

    # TODO: we only ever use this in visualizations.py; is this a problem??
    def get_subsets(self):
        return tf.cast(self.x_train_subset, dtype=tf.float32), tf.cast(self.y_train_subset, dtype=tf.float32), tf.cast(self.x_test_subset, dtype=tf.float32), tf.cast(self.y_test_subset, dtype=tf.float32)
    
    def generate_labeled_unlabeled_indices(self, data: np.ndarray, split_rate=0.5) -> tuple[np.ndarray, np.ndarray]:
        """ Randomly obtains indices of the data to have their labels kept or removed

        Args:
            data (np.ndarray): A features dataset
            split_rate (float, optional): The rate of labels to KEEP. Defaults to 0.5.

        Returns:
            tuple[np.ndarray, np.ndarray]: (labeled_indices, unlabeled_indices)
        """           
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
        """ Simple pre-processing step on the instance features dataset
        """        
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.

    def prepare_dataset(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, split_rate=0.5):
        """ Prepares and saves TF datasets corresponding to labeled_train_dataset, train_dataset, and test_dataset as instance variables
            Handles split_rate, which determines proportion of labeled/unlabeled data

        Args:
            x_train (np.ndarray): Features dataset for training
            y_train (np.ndarray): Labels dataset for training
            x_test (np.ndarray): Features dataset for testing
            y_test (np.ndarray): Labels dataset for testing
            split_rate (float, optional): The rate of labels to KEEP. Defaults to 0.5.
        """        
        # Labeled and unlabeled samples are loaded synchronously
        # with batch sizes selected accordingly
        # getting the indices for out labeled and unlabeled data
        train_x_labeled_idx, train_x_unlabeled_idx = self.generate_labeled_unlabeled_indices(x_train, split_rate=split_rate)
        labeled_dataset_size = len(train_x_labeled_idx)
        unlabeled_dataset_size = len(train_x_unlabeled_idx)
        
        steps_per_epoch = len(x_train) // batch_size
        # print(f'{steps_per_epoch=}')
        labeled_batch_size = labeled_dataset_size // steps_per_epoch
        unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
        # print(
        #     f"Batch size is: {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)"  
        # )

        # getting the unlable
        unlabeled_train_dataset = x_train[train_x_unlabeled_idx]
        unlabeled_train_dataset = (
            tf.data.Dataset.from_tensor_slices(unlabeled_train_dataset)\
            .shuffle(buffer_size=10*unlabeled_batch_size)
            .batch(unlabeled_batch_size))
        
        train_x_labeled = x_train[train_x_labeled_idx]
        train_y_labeled = y_train[train_x_labeled_idx]
        

        labeled_train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_x_labeled, train_y_labeled))
            .shuffle(buffer_size=10*labeled_batch_size)
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

    

    