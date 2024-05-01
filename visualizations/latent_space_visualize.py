import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
os.environ["KERAS_BACKEND"] = "tensorflow"

import resource
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#exit()
#import tensorflow_transform as tft
import keras
from keras import ops
from keras import layers
from model.augmentations import RandomColorAffine, get_augmenter
from model.contrastive_model import ContrastiveModel
from data.dataloader import Dataloader, download_data
from sklearn.decomposition import PCA


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

def reduce_PCA(features, n_dims=3):
    print(f'{features.shape=}')
    pca = PCA(n_dims)
    reduced_features = tf.transpose(pca.fit(tf.transpose(features)).components_)
    print(f'{reduced_features.shape=}, {n_dims=}')

    return reduced_features

def plot_3d(reduced_features, labels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    m = ['*', '+', 'D', '3', '4', 'X', 's', 'p', '8', 'P']
    for index in range(len(labels)):
        label = labels[index]
        #print("label is: " + str(label))
        cur_marker = m[int(label)]
        xs = reduced_features[index, 0]
        ys = reduced_features[index, 1]
        zs = reduced_features[index, 2]
        ax.scatter(xs, ys, zs, marker=cur_marker)

    """
    m = '1'
    xs = reduced_features[:10, 0]
    ys = reduced_features[:10, 1]
    zs = reduced_features[:10, 2]

    ax.scatter(xs, ys, zs, marker=m)

    m = '2'
    xs = reduced_features[10:20, 0]
    ys = reduced_features[10:20, 1]
    zs = reduced_features[10:20, 2]

    ax.scatter(xs, ys, zs, marker=m)
    """

    plt.show()
    plt.savefig("3d_latent_visualization.png")

model = load_model()
print()
print("Model Loaded")
print()
train_dataset, labeled_train_dataset, test_dataset = load_dataset()
features, labels = generate_latent_embeddings(model, test_dataset)
reduced_features = reduce_PCA(features)
plot_3d(reduced_features, labels)