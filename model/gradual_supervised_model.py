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

from collections import defaultdict


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

        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

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
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
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

        ########################################################################
        # BEGIN SEMI-SUPERVISED PORTION
        # Only labeled images are used WITH LABELS
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )

        # Use linear probe's classification loss to update both ENCODER and LINEAR PROBE
        # Since encoder is being updated with labels, this is SEMI-SUPERIVSED
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images, training=True)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
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
                                              num_classes_range=(8,10),
                                              split_rate_range=(0.01, 0.9),
                                              contrastive_learning_rate_range=(0.001,0.001),
                                              probe_learning_rate_range=(0.01,0.03))

    # setting the unique training rates for each part of the model
    gradual_supervised_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_contrastive_learning_rate),
        probe_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_probe_learning_rate),
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

    os.makedirs(r'../checkpoints', exist_ok=True)
    gradual_supervised_model.save_weights(f'../checkpoints/gradual_supervised_model_{title_acc}.weights.h5')
    print('Successfully saved model!')