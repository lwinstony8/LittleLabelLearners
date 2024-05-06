import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
os.environ["KERAS_BACKEND"] = "tensorflow"

# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers

from data.dataloader import Dataloader, download_data
from model.augmentations import get_augmenter
from model.custom_callbacks import ScheduledSubsetCallback
import model.hyperparameters as hp

import math
from collections import defaultdict
from itertools import repeat



# Define the contrastive model with model-subclassing
class GradualSupervised(keras.Model):
    def __init__(self, train: np.ndarray, test: np.ndarray, 
                 num_classes_range: int | tuple[int, int]=(5, 10), split_rate_range: float | tuple[float, float]=(0.5, 0.5),
                 contrastive_learning_rate_range: float | tuple[float, float]=(0.001,0.001),
                 probe_learning_rate_range: float | tuple[float, float]=(0.01,0.01)):
        """Initializer for GradualSupervised. Contains three modules: encoder, projection_head, and linear_probe

        Args:
            train (np.ndarray): Training dataset containing labeled and unlabeled samples
            test (np.ndarray): Testing dataset containing labeled and unlabeled samples
            num_classes_range (int | tuple[int, int], optional): INCLUSIVE range of possible subset_sizes. Use a single int to designate constant subset_size. Defaults to (5, 10).
            split_rate_range (float | tuple[float, float], optional): INCLUSIVE range of possible split_rates. Use a single float to designate constant split_rate. Defaults to (0.5, 0.5).
        """        

        super().__init__()
        self.temperature = hp.temperature
        self.contrastive_augmenter = get_augmenter(**hp.contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**hp.classification_augmentation)

        # both learning rates
        self.contrastive_learning_rate_range = contrastive_learning_rate_range if isinstance(contrastive_learning_rate_range, tuple) else (contrastive_learning_rate_range, contrastive_learning_rate_range)
        self.curr_contrastive_learning_rate = self.contrastive_learning_rate_range[1]
        self.probe_learning_rate_range = probe_learning_rate_range if isinstance(probe_learning_rate_range, tuple) else (probe_learning_rate_range, probe_learning_rate_range)
        self.curr_probe_learning_rate = self.probe_learning_rate_range[1]
        
        self.num_classes_range = num_classes_range if isinstance(num_classes_range, tuple) else (num_classes_range, num_classes_range)
        self.split_rate_range= split_rate_range if isinstance(split_rate_range, tuple) else (split_rate_range, split_rate_range)
        self.cur_num_classes = self.num_classes_range[0]
        self.cur_split_rate = self.split_rate_range[0]
        
        self.dataloader = Dataloader(train, test)
        self.dataloader.preprocess()
        self.dataloader.generate_subsets(subset_size=self.cur_num_classes) # Set base num_classes
        self.dataloader.prepare_dataset(
            self.dataloader.x_train_subset, 
            self.dataloader.y_train_subset, 
            self.dataloader.x_test, 
            self.dataloader.y_test,
            split_rate=self.cur_split_rate) # Set base split_rate

        self.encoder = keras.Sequential(
            [
                layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="leaky_relu"),
                layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="leaky_relu"),
                layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="leaky_relu"),
                layers.Conv2D(hp.width, kernel_size=3, strides=2, activation="leaky_relu"),
                layers.Flatten(),
                layers.Dense(hp.width, activation="leaky_relu"),
            ],
            name="encoder",
        )
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(hp.width,)),
                layers.Dense(hp.width, activation="leaky_relu"),
                layers.Dense(hp.width),
            ],
            name="projection_head",
        )
        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential([
            layers.Input(shape=(hp.width,)), 
            layers.Dense(10)],
            name="linear_probe",
        )

        self.pseudo_linear_probe = keras.Sequential([
            layers.Input(shape=(hp.width,)),
            layers.Dense(10)],
            name='pseudo_linear_probe')
        
        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, pseudo_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.pseudo_optimizer = pseudo_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

        self.pseudo_loss = keras.losses.SparseCategoricalCrossentropy()


    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = ops.normalize(projections_1, axis=1)
        projections_2 = ops.normalize(projections_2, axis=1)
        similarities = (
            ops.matmul(projections_1, ops.transpose(projections_2)) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = ops.shape(projections_1)[0]
        # tf.print(f'{batch_size=}')
        contrastive_labels = ops.arange(batch_size)
        # tf.print(f'{contrastive_labels=}')
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, ops.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, ops.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2
    
    def pseudo_classifier_soft_lookup(self, unlabeled_images: tf.Tensor, labeled_images: tf.Tensor, labels: tf.Tensor):
        """ A pseudo-classifier method using dot-product "soft-lookup" of unlabeled and labeled encodings
            to determine weighting.

            Experimental and not recommended for use!

        Args:
            unlabeled_images (tf.Tensor)
            labeled_images (tf.Tensor)
            labels (tf.Tensor)
        """        

        with tf.GradientTape(persistent=True) as tape:
            
            # Encode the images
            unlabeled_encodings = self.encoder(unlabeled_images, training=True)
            labeled_encodings = self.encoder(labeled_images, training=True)

            # Soft-lookup with QUERY=labeled_encodings; KEY=unlabeled_encodings
            attention = tf.matmul(labeled_encodings, unlabeled_encodings, transpose_b=True)/math.sqrt(hp.width)
            attention = tf.nn.softmax(attention)

            # Attention-valued unlabeled_encodings
            weighted_labeled_encodings = tf.matmul(attention, unlabeled_encodings)
            weighted_labeled_encodings = tf.nn.softmax(weighted_labeled_encodings)

            # Determine pseudo-labels of unlabeled encodings
            output = self.pseudo_linear_probe(weighted_labeled_encodings)

            # Compare to true labels
            pseudo_loss = self.pseudo_loss(labels, output)
            encoder_gradients = tape.gradient(pseudo_loss, self.encoder.trainable_weights)

            self.pseudo_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_weights))
    
    def pad_clusters(self, classes: list[tf.Tensor]) -> np.ndarray:
        """ Helper method for ensuring all classes have the same number of samples.
            Basically, creates duplicates of random existing samples to pad out a class

        Args:
            classes (list[tf.Tensor]): Intended to be a list of class-specific labeled encodings

        Returns:
            np.ndarray: Array of class-specific labeled encodings, where each class has the same number of labeled encodings
        """        

        acc = []
        max_size = max(len(c) for c in classes)

        # For each class, randomly choose existing labeled encodings to make duplicate (i.e. concatenating copies)
        # to pad out the class
        for c in classes:
            padding_idx = np.random.choice(np.arange(len(c)), size=(max_size - len(c)), replace=True)
            padding = c[padding_idx]
            acc.append(np.concatenate([c, padding]))
        
        return np.asarray(acc)
    
    def pseudo_classifier_reclassification_per_class(self, unlabeled_images: tf.Tensor, labeled_images: tf.Tensor, labels: tf.Tensor):
        """ Generate pseudo-labels for unlabeled encodings using an average similarity with respect to each class of 
            labeled encodings.

            These pseudolabels are then used to generate new labels for labeled encodings.
            Then, uses CCE to compare generated and true labels. These CCEs are used to update the encoder once PER CLASS (i.e. multiple gradient updates!)

            Intended to promote dense embeddings of unlabeled points around corresponding labeled points, 
            while pushing away unrelated points.

        Args:
            unlabeled_images (tf.Tensor)
            labeled_images (tf.Tensor)
            labels (tf.Tensor)

        """        

        def unlabeled_pseudo_labeler(unlabeled_encoding: tf.Tensor, distance_type='dot') -> tf.Tensor:
            """ Calculates similarity of given unlabeled encoding with respect to all classes of
                labeled encodings. Basically generates pseudolabels for given unlabeled encoding

            Args:
                unlabeled_encoding (tf.Tensor)
                distance_type (str, optional): Similarity type. Currently supporting inverse euclidean and dot product. Defaults to 'dot'.

            Returns:
                tf.Tensor: Pseudolabel of given unlabeled encoding
            """

            # Inverse Euclidean definition of similarity. Very slow!            
            if distance_type=='inv_euc':
                distances = tf.stack(
                    [tf.reduce_mean(
                        1/tf.reduce_sum(
                            (tf.broadcast_to(unlabeled_encoding, c.shape) - c) ** 2, axis=1)) for c in labeled_encodings_by_class], axis=0)
            
            # Dot product definition of similarity. Much faster
            else: 
                distances = tf.stack(tf.map_fn(
                    lambda c: tf.reduce_mean(
                        tf.einsum('j, ij->i', unlabeled_encoding, c)), labeled_encodings_by_class))
            distances = tf.nn.softmax(distances)

            return distances
        def labeled_pseudo_labeler(labeled_encoding: tf.Tensor, distance_type='dot') -> tf.Tensor:
            """ Generates labels for given labeled_encoding based on average similarity to all unlabeled encodings
                weighted by the unlabeled encodings' pseudolabels

            Args:
                labeled_encoding (tf.Tensor)
                distance_type (str, optional): Similarity type. Currently supporting inverse euclidean and dot product. Defaults to 'dot'.

            Returns:
                tf.Tensor: Pseudolabel of labeled encoding
            """

            # Inverse Euclidean definition of similarity. Very slow!            
            if distance_type=='inv_euc':
                distances = tf.stack([tf.reduce_mean(
                    1/tf.reduce_sum(
                        (tf.broadcast_to(tf.expand_dims(labeled_encoding, axis=0), c.shape) - c) ** 2, axis=-1)) for c in unlabeled_labeled_weighted_distances], axis=0)
            
            # Dot product definition of similarity. Faster.
            else:
                distances = tf.stack(tf.map_fn(lambda c: tf.reduce_mean(
                    tf.einsum('j, ij->i', labeled_encoding, c)), unlabeled_labeled_weighted_distances))
            distances = tf.nn.softmax(distances)
            return distances

        def updater(label_tuple: tf.Tensor) -> tf.Tensor:
            """ Calculates loss between generated and true labels for a given class.

            Args:
                label_tuple (tf.Tensor): (class-specific labeled_encodings, class_labels)

            Returns:
                tf.Tensor: Loss between given class of labeled encodings' pseudolabels and their true labels
            """            

            # Extract class-specific labeled_encodings and class_labels
            labeled_encodings, class_labels = label_tuple

            # Generate labeled_encodings' pseudolabels
            pseudo_predictions = tf.map_fn(labeled_pseudo_labeler, labeled_encodings)

            # CCE loss between generated pseudolabels and true labels
            pseudo_loss = self.pseudo_loss(class_labels, pseudo_predictions)

            return pseudo_loss
        
        # Cast all inputs; otherwise, tf.map_fn will throw a fit
        labels = tf.cast(labels, tf.int64)
        unlabeled_images = tf.cast(unlabeled_images, tf.float32)
        labeled_images = tf.cast(labeled_images, tf.float32)
        
        with tf.GradientTape() as tape:
            unlabeled_encodings = self.encoder(unlabeled_images, training=False) # [num_unlabeled, enc_sz]
            
            classes, _ = tf.unique(tf.squeeze(labels)) # [num_labeled]
            class_indices = tf.map_fn(lambda c: tf.where(labels==c)[0], classes)
            labels_by_class = tf.map_fn(lambda c_idx: tf.gather(labels, c_idx), class_indices)

            # Split labeled_images by class and then encode them
            labeled_images_by_class = tf.map_fn(lambda c_idx: tf.gather(labeled_images, c_idx), class_indices, 
                                                fn_output_signature=tf.float32)
            labeled_encodings_by_class = tf.map_fn(lambda c: self.encoder(c, training=True), labeled_images_by_class)
            
            # Find pseudolabels for unlabeled_encodings
            unlabeled_labeled_softmaxed_distances = tf.map_fn(unlabeled_pseudo_labeler, unlabeled_encodings) # [num_unlabeled, num_classes]

            # Weigh each unlabeled point by pseudolabels
            # [num_classes, enc_sz, num_unlabeled]
            unlabeled_labeled_weighted_distances = tf.einsum('ij, ik -> kij', 
                                                             unlabeled_encodings, 
                                                             unlabeled_labeled_softmaxed_distances)

            # Accumulate loss between generated pseudolabels and true labels for labeled encodings
            pseudo_loss_list = tf.map_fn(updater, (labeled_encodings_by_class, labels_by_class), fn_output_signature=tf.float32)

        # Find gradients for each class' pseudolabels, and then apply them PER CLASS
        encoder_gradients = tf.map_fn(lambda loss: tape.gradient(loss, self.encoder.trainable_weights), pseudo_loss_list)
        tf.map_fn(lambda grad_tuple: self.pseudo_optimizer.apply_gradients(grad_tuple), (encoder_gradients, repeat(self.encoder.trainable_weights)))

    def pseudo_classifier_reclassification_weighted(self, unlabeled_images: tf.Tensor, labeled_images: tf.Tensor, labels: tf.Tensor):
        """ Generate pseudo-labels for unlabeled encodings using an average similarity with respect to each class of 
            labeled encodings.

            These pseudolabels are then used to generate new labels for labeled encodings.
            Then, uses CCE to compare generated and true labels. These CCEs are used to update the encoder AT ONCE (i.e. single gradient update!)

            Intended to promote dense embeddings of unlabeled points around corresponding labeled points, 
            while pushing away unrelated points.

        Args:
            unlabeled_images (tf.Tensor)
            labeled_images (tf.Tensor)
            labels (tf.Tensor)

        """    

        def unlabeled_pseudo_labeler(unlabeled_encoding: tf.Tensor, distance_type='dot') -> tf.Tensor:
            """ Calculates similarity of given unlabeled encoding with respect to all classes of
                labeled encodings. Basically generates pseudolabels for given unlabeled encoding

            Args:
                unlabeled_encoding (tf.Tensor)
                distance_type (str, optional): Similarity type. Currently supporting inverse euclidean and dot product. Defaults to 'dot'.

            Returns:
                tf.Tensor: Pseudolabel of given unlabeled encoding
            """

            # Inverse Euclidean definition of similarity. Very slow!            
            if distance_type=='inv_euc':
                distances = tf.stack(
                    [tf.reduce_mean(
                        1/tf.reduce_sum(
                            (tf.broadcast_to(unlabeled_encoding, c.shape) - c) ** 2, axis=1)) for c in labeled_encodings_by_class], axis=0)
            
            # Dot product definition of similarity. Much faster
            else: 
                distances = tf.stack(tf.map_fn(
                    lambda c: tf.reduce_mean(
                        tf.einsum('j, ij->i', unlabeled_encoding, c)), labeled_encodings_by_class))
            distances = tf.nn.softmax(distances)

            return distances
        def labeled_pseudo_labeler(labeled_encoding: tf.Tensor, distance_type='dot') -> tf.Tensor:
            """ Generates labels for given labeled_encoding based on average similarity to all unlabeled encodings
                weighted by the unlabeled encodings' pseudolabels

            Args:
                labeled_encoding (tf.Tensor)
                distance_type (str, optional): Similarity type. Currently supporting inverse euclidean and dot product. Defaults to 'dot'.

            Returns:
                tf.Tensor: Pseudolabel of labeled encoding
            """

            # Inverse Euclidean definition of similarity. Very slow!            
            if distance_type=='inv_euc':
                distances = tf.stack([tf.reduce_mean(
                    1/tf.reduce_sum(
                        (tf.broadcast_to(tf.expand_dims(labeled_encoding, axis=0), c.shape) - c) ** 2, axis=-1)) for c in unlabeled_labeled_weighted_distances], axis=0)
            
            # Dot product definition of similarity. Faster.
            else:
                distances = tf.stack(tf.map_fn(lambda c: tf.reduce_mean(
                    tf.einsum('j, ij->i', labeled_encoding, c)), unlabeled_labeled_weighted_distances))
            distances = tf.nn.softmax(distances)
            return distances

        def updater(label_tuple: tf.Tensor) -> tf.Tensor:
            """ Calculates loss between generated and true labels for a given class. Weighs this loss by proportional class representation
                 (i.e. a popular class should contribute more to cumulative loss value)

            Args:
                label_tuple (tf.Tensor): (class-specific labeled_encodings, class_labels)

            Returns:
                tf.Tensor: Loss between given class of labeled encodings' pseudolabels and their true labels, weighted proportinally by class representation
            """ 


            # Extract class-specific labeled_encodings and class_labels
            labeled_encodings, class_labels = label_tuple

            # Generate labeled_encodings' pseudolabels
            pseudo_predictions = tf.map_fn(labeled_pseudo_labeler, labeled_encodings)

            # CCE loss between generated pseudolabels and true labels
            pseudo_loss = self.pseudo_loss(class_labels, pseudo_predictions)

            # Weigh pseudo_loss by class representation proportion
            loss_weight = class_labels.shape[0] / self.cur_num_classes
            return tf.cast(loss_weight, tf.float32)*pseudo_loss
        
        # Cast all inputs; otherwise, tf.map_fn will throw a fit
        labels = tf.cast(labels, tf.int64)
        unlabeled_images = tf.cast(unlabeled_images, tf.float32)        
        labeled_images = tf.cast(labeled_images, tf.float32)

        with tf.GradientTape() as tape:
            unlabeled_encodings = self.encoder(unlabeled_images, training=True) # [num_unlabeled, enc_sz]

            classes, _ = tf.unique(tf.squeeze(labels)) # [num_labeled]
            class_indices = tf.map_fn(lambda c: tf.where(labels==c)[0], classes)
            labels_by_class = tf.map_fn(lambda c_idx: tf.gather(labels, c_idx), class_indices)

            # Split labeled_images by class and then encode them
            labeled_images_by_class = tf.map_fn(lambda c_idx: tf.gather(labeled_images, c_idx), class_indices, 
                                                fn_output_signature=tf.float32)
            labeled_encodings_by_class = tf.map_fn(lambda c: self.encoder(c, training=True), labeled_images_by_class)

            # Find pseudolabels for unlabeled_encodings
            unlabeled_labeled_softmaxed_distances = tf.map_fn(unlabeled_pseudo_labeler, unlabeled_encodings)

            # Weigh each unlabeled point by pseudolabels
            # [num_classes, enc_sz, num_unlabeled]
            unlabeled_labeled_weighted_distances = tf.einsum('ij, ik -> kij', 
                                                             unlabeled_encodings, 
                                                             unlabeled_labeled_softmaxed_distances)

            # Calculate weighted cumulative loss between generated pseudolabels and true labels for labeled encodings
            pseudo_loss_cumsum = tf.map_fn(updater, (labeled_encodings_by_class, labels_by_class), fn_output_signature=tf.float32)
            pseudo_loss_cumsum = tf.reduce_sum(pseudo_loss_cumsum)

        # Using cumulative weighted loss, apply gradient update to encoder's weights
        encoder_gradient = tape.gradient(pseudo_loss_cumsum, self.encoder.trainable_weights)
        self.pseudo_optimizer.apply_gradients(zip(encoder_gradient, self.encoder.trainable_weights))
    
    def meta_classifier_reclassification(self, unlabeled_images: tf.Tensor, labeled_images: tf.Tensor, labels: tf.Tensor):
        """ Generates "meta" representations of labeled encodings, which are then used in predictions. Loss between metapoints' labels and true labels
            is used to update encoder AND pseudo_linear_probe

            Essentially, we first find the similarity between unlabeled and labeled points. Then, we generate "meta" points by finding similarity of
            labeled points and unlabeled points, weighted by unlabeled points' similarities. We can think of this as labeled points 
            under gravitational pull of unlabeled points, whose pull is weighted by unlabeled points' similarities.

            Introduces new parameters to model; not recommended. Instead, consider meta_classifier_distance

        Args:
            unlabeled_images (tf.Tensor)
            labeled_images (tf.Tensor)
            labels (tf.Tensor)
        """        
        with tf.GradientTape(persistent=True) as tape:
            unlabeled_encodings = self.encoder(unlabeled_images, training=False) #[473, 128]
            labeled_encodings = self.encoder(labeled_images, training=True) #[52, 128]

            # Similarity between labeled and unlabeled points
            unlabeled_labeled_dots = tf.einsum('ij, kj -> ik', unlabeled_encodings, labeled_encodings) #[473, 52]
            # unlabeled_labeled_dots = tf.nn.softmax(unlabeled_labeled_dots) # Is this needed?
            
            # Weigh the unlabeled points by the similarity
            weighted_unlabeled_encodings = tf.einsum('ij, ik -> jik', unlabeled_labeled_dots, unlabeled_encodings) #[52, 473, 128]; [num_labeled, num_unlabeled, enc_sz]

            # For each labeled point, find weighted average to all unlabeled points
            # This transforms each labeled point into a "meta" point
            pseudo_predictions = tf.einsum('ijk, ik -> ik', weighted_unlabeled_encodings, labeled_encodings) #[52, 128]
            
            # Uses new pseudo_linear_probe to calculate pseudo_predictions. Can also consider using linear_probe...
            pseudo_predictions = self.pseudo_linear_probe(pseudo_predictions, training=True)
            # pseudo_predictions = self.linear_probe(pseudo_predictions, training=True)

            pseudo_loss = self.pseudo_loss(labels, pseudo_predictions)

        # Update both encoder and pseudo_linear_probe
        encoder_gradients = tape.gradient(pseudo_loss, self.encoder.trainable_weights)
        self.contrastive_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_weights))
        
        pseudo_linear_probe_gradients = tape.gradient(pseudo_loss, self.pseudo_linear_probe.trainable_weights)
        self.pseudo_optimizer.apply_gradients(zip(pseudo_linear_probe_gradients, self.pseudo_linear_probe.trainable_weights))
    
    def meta_classifier_distance(self, unlabeled_images: tf.Tensor, labeled_images: tf.Tensor):
        """ An encoder-updating method using dot products between unlabeled and labeled encodings
            to create "meta"-labeled points. Minimize the distance between each labeled encoding and its
            meta variant.

            Improves contrastive performance compared to plain GradualSupervised()

        Args:
            unlabeled_images (tf.Tensor)
            labeled_images (tf.Tensor)
        """
        
        with tf.GradientTape() as tape:

            # Obtain images' encodings
            unlabeled_encodings = self.encoder(unlabeled_images, training=False) # [num_unlabeled, enc_sz]
            labeled_encodings = self.encoder(labeled_images, training=True) # [num_labeled, enc_sz]

            # Similarity between labeled and unlabeled points
            unlabeled_labeled_dots = tf.einsum('ij, kj -> ik', unlabeled_encodings, labeled_encodings) #[num_unlabeled, num_labeled]
            unlabeled_labeled_dots = tf.nn.softmax(unlabeled_labeled_dots) 
            
            # Weigh the unlabeled points by the similarity
            weighted_unlabeled_encodings = tf.einsum('ij, ik -> jik', unlabeled_labeled_dots, unlabeled_encodings) #[num_labeled, num_unlabeled, enc_sz]

            # For each labeled point, find weighted average to all unlabeled points
            # This transforms each labeled point into a "meta" point
            pseudo_points = tf.einsum('ijk, ik -> ik', weighted_unlabeled_encodings, labeled_encodings) #[52, 128]

            # Find average log euc-distance between each labeled_point and its meta counterpart
            pseudo_loss = -tf.reduce_mean(tf.math.log(tf.math.sqrt(tf.reduce_sum((pseudo_points-labeled_encodings)**2, axis=-1))))

        # Update encoder's weights
        encoder_gradients = tape.gradient(pseudo_loss, self.encoder.trainable_weights)
        self.contrastive_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_weights))

    def train_step(self, data: tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]) -> dict[str, tf.float32]:
        """ Runs the training routine for a single step; i.e. trains on a single batch
            Training routine consists of two steps:
                1. Use both labeled and unlabeled images to update encoder AND projection heads in a self-supervised manner (i.e. NO LABELS)
                2. Use only labeled images to update encoder AND linear probe in a semi-supervised manner (i.e. WITH LABELS)
            
            Note, step 2 differentiates GradualSupervisedModel from SelfSupervisedModel in that for the latter, labeled images do NOT update the encoder

        Args:
            data (tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]): (unlabeled_images, (labeled_images, labels))

        Returns:
            dict[str, tf.float32]: metrics
        """        
        unlabeled_images = data[0]
        labeled_images, labels = data[1]

        ########################################################################
        # SELF-SUPERVISED PORTION
        # Both labeled and unlabeled images are used WITHOUT LABELS
        images = ops.concatenate((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)

        # Use contrastive_loss to update encoder WITHOUT LABELS
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1,projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        # END SELF-SUPERVISED PORTION
        ########################################################################
        # BEGIN SEMI-SUPERVISED PORTION
        
        ########################################################################
        ## OPTIONAL PSEUDO/META CLASSIFIER-BASED ENCODER UPDATES
        # self.pseudo_classifier_soft_lookup(unlabeled_images, labeled_images, labels)
        # self.pseudo_classifier_reclassification_per_class(unlabeled_images, labeled_images, labels)
        # self.pseudo_classifier_reclassification_weighted(unlabeled_images, labeled_images, labels)
        # self.meta_classifier_reclassification(unlabeled_images, labeled_images, labels)
        # self.meta_classifier_distance(unlabeled_images, labeled_images)
        ########################################################################
        
        # Only labeled images are used WITH LABELS
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )

        # Use linear probe's classification loss to update both ENCODER and LINEAR PROBE
        # Since encoder is being updated with labels, this is SEMI-SUPERIVSED

        # Without training encoder
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        probe_gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(probe_gradients, self.linear_probe.trainable_weights)
        )
        
        '''
        # Train linear_probe AND encoder
        with tf.GradientTape(persistent=True) as tape:
            features = self.encoder(preprocessed_images, training=True)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        probe_gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        encoder_gradients = tape.gradient(probe_loss, self.encoder.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(probe_gradients, self.linear_probe.trainable_weights)
        )
        self.contrastive_optimizer.apply_gradients(
            zip(encoder_gradients, self.encoder.trainable_weights)
        )
        del tape
        '''
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        # END SEMI-SUPERVISED PORTION
        ########################################################################

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
    
if __name__ == '__main__':
    train, test = download_data()

    # Contrastive pretraining
    gradual_supervised_model = GradualSupervised(train, test,
                                              num_classes_range=10,
                                              split_rate_range=(0.1, 0.9),
                                              contrastive_learning_rate_range=0.01,
                                              probe_learning_rate_range=(0.01,0.03))

    # setting the unique training rates for each part of the model
    gradual_supervised_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_contrastive_learning_rate),
        pseudo_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_contrastive_learning_rate),
        probe_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_probe_learning_rate),
        # run_eagerly=True # TODO: Uncomment if debugging
    )
    
    model_history = defaultdict(lambda: [])
    scheduled_subset_callback = ScheduledSubsetCallback(gradual_supervised_model)
    for epoch in range(hp.num_epochs):
        scheduled_subset_callback(cur_epoch=epoch)
        for k, v in gradual_supervised_model.fit(
            gradual_supervised_model.dataloader.train_dataset, 
            epochs=1, # NOTE: this has to be 1; we only use each subset ONCE
            validation_data=gradual_supervised_model.dataloader.test_dataset,
        ).history.items():
            model_history[k].extend(v)

    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(model_history["val_p_acc"]) * 100
        )
    )
    title_acc = "{:.2f}".format(max(model_history["val_p_acc"]) * 100)

    # os.makedirs(r'../checkpoints', exist_ok=True)
    # gradual_supervised_model.save_weights(f'../checkpoints/gradual_pseudo_supervised_model_{title_acc}.weights.h5')
    # print('Successfully saved model!')