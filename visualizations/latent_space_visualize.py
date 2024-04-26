import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
os.environ["KERAS_BACKEND"] = "tensorflow"

import resource
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
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
    # tft.pca(features, 4)
    pca = PCA(n_dims)
    reduced_features = tf.transpose(pca.fit(tf.transpose(features)).components_)
    print(f'{reduced_features.shape=}, {n_dims=}')

    return reduced_features



model = load_model()
print()
print("Model Loaded")
print()
train_dataset, labeled_train_dataset, test_dataset = load_dataset()
features, labels = generate_latent_embeddings(model, test_dataset)
reduced_features = reduce_PCA(features, 2)

def get_n_colors(n):
    colormap = plt.cm.get_cmap('tab10', n)  # You can change 'tab10' to other colormaps
    colors = [colormap(i) for i in range(n)]
    color_dict = {i: colors[i] for i in range(n)}
    return color_dict

num_labels = len(np.unique(labels))
# print(f'{labels=}')
# print(f'{num_labels=}')
# color_dict = get_n_colors(num_labels)

# for label in np.unique(labels):
#     x = reduced_features[:,0]
#     y = reduced_features[:,1]
#     labels_idx = np.nonzero(labels == label)[0]
#     plt.scatter(x[labels_idx], y[labels_idx], colors=color_dict[label], label=f'Label {label}')

cm = plt.scatter(reduced_features[:,0], reduced_features[:,1], c=tf.squeeze(labels))
plt.colorbar(cm)
plt.show()
plt.savefig('visualize_latent_pca.png')

# print(f'{labels.shape=}')
'''
# UMAP
import umap
mapper = umap.UMAP().fit(features)
import umap.plot
p = umap.plot.points(mapper, labels=tf.squeeze(labels), s=20)
umap.plot.plt.show()
umap.plot.plt.savefig('visualize_latent_umap.png')
'''