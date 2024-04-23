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
my_dataloader = Dataloader(train, test)

my_dataloader.preprocess()
my_dataloader.generate_subsets()

# y_train_onehot, y_test_onehot = my_dataloader.one_hot(my_dataloader.y_train, my_dataloader.y_test)

# Load STL10 dataset
train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
    my_dataloader.x_train, 
    my_dataloader.y_train, 
    my_dataloader.x_test, 
    my_dataloader.y_test)

# print(f'{labeled_train_dataset=}')
# print(f'{test_dataset=}')
# exit()

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
        # get_augmenter(**classification_augmentation),
        get_encoder(),
        layers.Dense(10),
    ],
    name="baseline_model",
)
baseline_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

baseline_history = baseline_model.fit(
    labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(baseline_history.history["val_acc"]) * 100
    )
)