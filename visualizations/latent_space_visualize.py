import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
os.environ["KERAS_BACKEND"] = "tensorflow"

import resource
import math
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_transform as tft
import keras
from keras import ops
from keras import layers
from model.augmentations import RandomColorAffine, get_augmenter
from model.contrastive_model import ContrastiveModel
from data.dataloader import Dataloader, download_data




low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def load_model():
    model = ContrastiveModel()
    model.load_weights('../checkpoints/pretraining_model.weights.h5')
    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
    )
    return model

def load_dataset():
    train, test = download_data()
    my_dataloader = Dataloader(train, test)

    my_dataloader.preprocess()
    my_dataloader.generate_subsets()

    train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
        my_dataloader.x_train, 
        my_dataloader.y_train, 
        my_dataloader.x_test, 
        my_dataloader.y_test)
    return train_dataset, labeled_train_dataset, test_dataset

def generate_latent_embeddings(model, data):

    features = None
    for batch_inputs, batch_labels in data:
        preprocessed_images = model.classification_augmenter(
            batch_inputs, training=False
        )
        features = model.encoder(preprocessed_images, training=False)
        break
    
    return features, batch_labels

def reduce_features_3d(features):
    print("features shape: " + str(features.shape))
    tft.pca(features, 4)
    print("features shape2: " + str(features.shape))

    return reduced_features



model = load_model()
print()
print("Model Loaded")
print()
train_dataset, labeled_train_dataset, test_dataset = load_dataset()
features, labels = generate_latent_embeddings(model, test_dataset)
reduce_features_3d(features)