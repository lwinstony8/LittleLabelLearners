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

train, test = download_data()
my_dataloader = Dataloader(train, test)


# Dataset hyperparameters
unlabeled_dataset_size = 100000
labeled_dataset_size = 5000
image_channels = 3

# Algorithm hyperparameters
num_epochs = 20
batch_size = 525  # Corresponds to 200 steps per epoch
width = 128
temperature = 0.1
# Stronger augmentations for contrastive, weaker ones for supervised training
contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {
    "min_area": 0.75,
    "brightness": 0.3,
    "jitter": 0.1,
}

def prepare_dataset():
    # Labeled and unlabeled samples are loaded synchronously
    # with batch sizes selected accordingly
    steps_per_epoch = (unlabeled_dataset_size + labeled_dataset_size) // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
    labeled_batch_size = labeled_dataset_size // steps_per_epoch
    print(
        f"batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)"
    )

    # getting our preprocessed data
    train, test = download_data()
    my_dataloader = Dataloader(train, test)
    my_dataloader.preprocess()
    my_dataloader.generate_subsets()    

    # getting the indices for out labeled and unlabeled data
    train_x_labeled_idx, train_x_unlabeled_idx = my_dataloader.generate_labeled_unlabeled_indices(my_dataloader.x_train)

    # getting the unlable
    unlabeled_train_dataset = my_dataloader.x_train[train_x_unlabeled_idx]
    unlabeled_train_dataset = (
        tf.data.Dataset.from_tensor_slices(unlabeled_train_dataset)\
        # .shuffle()
        .batch(batch_size))
    
    #
    train_x_labeled = my_dataloader.x_train[train_x_labeled_idx]
    train_y_labeled = my_dataloader.y_train[train_x_labeled_idx]
    labeled_train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_x_labeled, train_y_labeled))
        # .shuffle()
        .batch(batch_size)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((my_dataloader.x_test, my_dataloader.y_test))
        # .shuffle()
        .batch(batch_size)
    )

    # Labeled and unlabeled datasets are zipped together
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    )
    # prefetch if we need to

    '''
    # Turning off shuffle to lower resource usage
    unlabeled_train_dataset = (
        tfds.load("stl10", split="unlabelled", as_supervised=True, shuffle_files=False)
        .shuffle(buffer_size=10 * unlabeled_batch_size)
        .batch(unlabeled_batch_size)
    )
    labeled_train_dataset = (
        tfds.load("stl10", split="train", as_supervised=True, shuffle_files=False)
        .shuffle(buffer_size=10 * labeled_batch_size)
        .batch(labeled_batch_size)
    )
    test_dataset = (
        tfds.load("stl10", split="test", as_supervised=True)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    
    '''

    # train_dataset in tuple of itself to match
    return train_dataset, labeled_train_dataset, test_dataset


# Load STL10 dataset
train_dataset, labeled_train_dataset, test_dataset = prepare_dataset()

def visualize_augmentations(num_images):
    # Sample a batch from a dataset
    next_ds = next(iter(train_dataset))
    images = next_ds[0][:num_images]

    print(f'{images.shape=}')

    # Apply augmentations
    augmented_images = zip(
        images,
        get_augmenter(**classification_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
    )
    row_titles = [
        "Original:",
        "Weakly augmented:",
        "Strongly augmented:",
    ]
    plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(3, num_images, row * num_images + column + 1)
            plt.imshow(image)
            plt.savefig('figure.png')
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")
    plt.tight_layout()


# visualize_augmentations(num_images=8)


# Define the encoder architecture
def get_encoder():
    return keras.Sequential(
        [
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    )

# Baseline supervised training with random initialization
baseline_model = keras.Sequential(
    [
        get_augmenter(**classification_augmentation),
        get_encoder(),
        layers.Dense(10),
    ],
    name="baseline_model",
)
baseline_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

baseline_history = baseline_model.fit(
    labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(baseline_history.history["val_acc"]) * 100
    )
)