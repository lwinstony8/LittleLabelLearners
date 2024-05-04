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
import numpy as np

import keras
from keras import ops
from keras import layers
from model.augmentations import RandomColorAffine, get_augmenter
from model.contrastive_model import ContrastiveModel
from model.gradual_supervised_model import GradualSupervised
from data.dataloader import Dataloader, download_data

from sklearn.decomposition import PCA


from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import KNeighborsClassifier
import umap
import umap.plot


low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def load_model(model_type: str= 'GradualSupervised', weight_path: str=r'../checkpoints/pretraining_model.weights.h5') -> keras.Model:    
    """ Instantiates a Model, loads it with previously saved model_weights, compiles it, and then returns the compiled model

    Args:
        model_type (str, optional): str name of Model type. Defaults to 'GradualSupervised'.
        weight_path (str, optional): Path to previously saved model weights. Defaults to r'../checkpoints/pretraining_model.weights.h5'.

    Returns:
        keras.Model: Compiled model using previously saved weights
    """    
    
    # model = ContrastiveModel()
    train, test = download_data()
    if model_type == 'GradualSupervised':
        model = GradualSupervised(train, test)

    model.load_weights(weight_path)
    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
    )
    return model

def load_dataset(model: keras.Model) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """ Processes given model's dataloader's data and returns datasets

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: (Dataloader.train_dataset, Dataloader.labeled_train_dataset, Dataloader.test_dataset)
    """    

    model.dataloader.preprocess()
    model.dataloader.generate_subsets()

    model.dataloader.prepare_dataset(
        model.dataloader.x_train, 
        model.dataloader.y_train, 
        model.dataloader.x_test, 
        model.dataloader.y_test)
    return model.dataloader.train_dataset, model.dataloader.labeled_train_dataset, model.dataloader.test_dataset

def generate_latent_embeddings(model: keras.Model, data) -> tuple[tf.Tensor, tf.Tensor]:
    """ Generates and returns the latent embeddings calculated by the given model's encoder

    Args:
        model (keras.Model): Given model; should have an encoder layer as an instance variable
        data (_type_): Intended to be a batched tf.Data.dataset

    Returns:
        tuple[tf.Tensor, tf.Tensor]: (encoded_features_acc, labels_acc)
    """    

    encoded_features_acc = []
    labels_acc = []
    for batch_inputs, batch_labels in data:
        preprocessed_images = model.classification_augmenter(
            batch_inputs, training=False
        )
        encoded_features_acc.append(model.encoder(preprocessed_images, training=False))
        labels_acc.append(batch_labels)
    
    # print(f'{len(features_acc)=}')
    # print(f'{len(labels_acc)=}')
    encoded_features_acc = tf.concat(encoded_features_acc, axis=0)
    labels_acc = tf.concat(labels_acc, axis=0)

    #print(f'{encoded_features_acc.shape=}')
    #print(f'{labels_acc.shape=}')
    
    return encoded_features_acc, labels_acc

def generate_cluster_labels(encoded_data: tf.Tensor, num_clusters: int=10) -> np.ndarray:
    """ Via KMeans, generate and return num_clusters cluster labels using the given encoded_data

    Args:
        encoded_data (tf.Tensor): Encoded data; intended to be the product of calling a model's encoding layer on some batch of images
        num_clusters (int, optional): Number of clusters to generate. Defaults to 10.

    Returns:
        np.ndarray: Each encoded_data point's KMeans-assigned cluster label
    """    
    # fitting the data into clusters (will be compared to what our model outputted)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(encoded_data)
    return kmeans.labels_

def generate_KNN_predictions(encoded_data: tf.Tensor, true_labels: tf.Tensor, split_rate=0.8) -> tuple[np.ndarray, np.ndarray]:
    """ Splits given encoded_data and true_labels by split_rate. uses train data/labels to fit a KNNClassifier. 
        Using fit classifier, predicts (and thus generates labels) on test_data.

    Args:
        encoded_data (tf.Tensor): Encoded data; intended to be the product of calling a model's encoding layer on some batch of images
        true_labels (tf.Tensor): True labels
        split_rate (float, optional): Proportion of data to keep as train. Defaults to 0.8.

    Returns:
        tuple[np.ndarray, np.ndarray]: (KNN_preds, test_true_labels)
    """    
    # using kneighbors train a classifier, splitting the data into training and test, 
    # and then use the true_labels to see how good it is
    train_encoded_data = encoded_data[:int(len(encoded_data)*split_rate)]
    train_true_labels = true_labels[:int(len(true_labels)*split_rate)]

    test_encoded_data = encoded_data[int(len(encoded_data)*split_rate):]
    test_true_labels = true_labels[int(len(true_labels)*split_rate):]
    
    # can change the number of neighbors considered
    #print(f'{train_encoded_data.shape=}')
    #print(f'{train_true_labels.shape=}')
    neighbors = KNeighborsClassifier(n_neighbors=5)
    neighbors.fit(train_encoded_data, train_true_labels)

    KNN_preds = neighbors.predict(test_encoded_data)
    #nmi/ari?

    return KNN_preds, test_true_labels

def reduce_PCA(features: tf.Tensor, n_dims: int=3) -> tf.Tensor:
    """ Via PCA, returns a reduced_features that only has n_dims dimensions

    Args:
        features (tf.Tensor): Given dataset; intended to be the encoding latent-space representation
        n_dims (int, optional): Number of final dimensions/components to keep. Defaults to 3.

    Returns:
        tf.Tensor: reduced_features representation of original tensor, containing only n_dims dimensions
    """    
    # print(f'{type(features)=}')
    print('Running reduce_PCA')
    pca = PCA(n_dims)
    reduced_features = tf.transpose(pca.fit(tf.transpose(features)).components_)
    # print(f'{features.shape=}')
    # print(f'{reduced_features.shape=}, {n_dims=}')

    return reduced_features


def plot_3d(features, labels):
    reduced_features = reduce_PCA(features, 3)
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
    plt.savefig("3d_latent_visualization_gradual.png")

def visualize_PCA(features: tf.Tensor, labels: tf.Tensor):
    """ Given features and labels, generate and save a scatterplot

    Args:
        features (tf.Tensor): Intended to be the output of a model's encoder layer on a batch of images
        labels (tf.Tensor): Labels associated with data; can be true labels, KMeans-assigned labels, KNN labels, etc. Used for coloring scatter-plot
    """    
    print('Running visualize_PCA')
    reduced_features = reduce_PCA(features, 2)
    cm = plt.scatter(reduced_features[:,0], reduced_features[:,1], c=tf.squeeze(labels))
    plt.colorbar(cm)
    plt.show()
    plt.savefig('visualize_latent_pca_gradual.png')

def visualize_UMAP(features: tf.Tensor, labels: tf.Tensor):
    """ Given features and labels, generate and save UMAP dimensionality-reduced scatter-plot

    Args:
        features (tf.Tensor): Intended to be the output of a model's encoder layer on a batch of images
        labels (tf.Tensor): Labels associated with data; can be true labels, KMeans-assigned labels, KNN labels, etc. Used for coloring scatter-plot
    """    
    print('Running visualize_UMAP')
    mapper = umap.UMAP().fit(features)
    p = umap.plot.points(mapper, labels=tf.squeeze(labels))
    umap.plot.plt.show()
    umap.plot.plt.savefig('visualize_latent_umap_gradual.png')
    
def generate_scores(encoded_features: tf.Tensor, true_labels: tf.Tensor):
    """ Given the encoded features and true labels, calculates the different scores that quantify our clusters

    Args:
        encoded_features (tf.Tensor): Intended to be the output of a model's encoder layer on a batch of images; latent-space representation
        true_labels (tf.Tensor): True labels of original images; must have same batch_size (0th dimension) as encoded_featuers
    """    

    encoded_labels = generate_cluster_labels(encoded_features)
    #print(f'{encoded_labels.shape=}')

    kmeans_silhouette = silhouette_score(encoded_features, encoded_labels)
    true_silhouette = silhouette_score(encoded_features, true_labels)
    print(f'{true_silhouette=}')


    # calculate the normalized mutual information using the true labels and our predictions
    # gives us another measure of how good our clusterings are numerically
    # NOTE: from sklearn, "This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way."
    nmi_score = normalized_mutual_info_score(labels_true=true_labels[:,0], labels_pred=encoded_labels)
    print(f'{nmi_score=}')

    # calculate the adjusted rand index using the true labels and our predicted labels 
    # gives us a measure of how good our clusterings are numerically
    KNN_preds, test_true_labels = generate_KNN_predictions(encoded_features, true_labels[:,0])
    a_rand_score = adjusted_rand_score(labels_true=KNN_preds, labels_pred=test_true_labels)
    print(f'{a_rand_score=}')

def main():
    ''' Main method of the file called when th fle s run '''
    # create the model instance and 
    # model = GradualSupervised()
    model = load_model(model_type='GradualSupervised', weight_path=r'../checkpoints/gradual_pseudo_supervised_model_29.11.weights.h5')
    print("Model Loaded")
    train_dataset, labeled_train_dataset, test_dataset = load_dataset(model)
    encoded_features, true_labels = generate_latent_embeddings(model, test_dataset)
    # print(f'{true_labels.shape=}')
    # print(f'{len(test_dataset)=}')
    # test_labels = np.concatenate([y for _, y in test_dataset], axis=0)
    #print(f'{test_labels.shape=}')
    # generate_scores(encoded_features=encoded_features,true_labels=true_labels)
    visualize_PCA(encoded_features, true_labels)
    visualize_UMAP(encoded_features, true_labels)
    # plot_3d(encoded_features, true_labels)



if __name__=="__main__":
    main()

def get_n_colors(n):
    colormap = plt.cm.get_cmap('tab10', n)  # You can change 'tab10' to other colormaps
    colors = [colormap(i) for i in range(n)]
    color_dict = {i: colors[i] for i in range(n)}
    return color_dict

# num_labels = len(np.unique(true_labels))
# print(f'{labels=}')
# print(f'{num_labels=}')
# color_dict = get_n_colors(num_labels)

# for label in np.unique(labels):
#     x = reduced_features[:,0]
#     y = reduced_features[:,1]
#     labels_idx = np.nonzero(labels == label)[0]
#     plt.scatter(x[labels_idx], y[labels_idx], colors=color_dict[label], label=f'Label {label}')

# print(f'{labels.shape=}')