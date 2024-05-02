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
    
    def pseudo_classified(self, unlabeled_images, labeled_images, labels):
        """ In the encoder's embedding space, propagate via labeled data pseudo-labels to unlabeled data.
            Then, using pseudo-labeled data, generate label probabilities for labeled data.
            Finally, CCE between pseudo-classified data and true labeled_data
            
            1. Take in embedded data and save into a "memory bank" what is labeled/unlabeled
            
            2. For each unlabeled datapoint:
                -   calculate the average distance to each labeled data point ACROSS classes 
                    i.e. {0: avg_dist0, 1: avg_dist1, ...}. Update memory bank
                -   softmax ACROSS classes
                -   argmax to find pseudolabel (i.e. class)
                -   update memory bank

            3. For each labeled datapoint:
                -   labeled_sum_acc = {class0: 0, class1: 0, class2: 0 ...}
                -   For each unlabeled datapoint:
                    -   sum
        
        """        

        with tf.GradientTape(persistent=True) as tape:
            unlabeled_encodings = self.encoder(unlabeled_images, training=True)
            labeled_encodings = self.encoder(labeled_images, training=True)

            # Q = self.w_Q(labeled_encodings, training=True)
            # K = self.w_K(unlabeled_encodings, training=True)
            # V = self.w_V(unlabeled_encodings, training=True)

            attention = tf.matmul(labeled_encodings, unlabeled_encodings, transpose_b=True)/math.sqrt(hp.width)
            attention = tf.nn.softmax(attention)

            # calculate the weighted_labeled_encodings
            # weigh our unlabeled encodings based on how well it relates
            # using the attention to "label" the 
            #print(f'{attention.shape=}')
            '''
            weighted_labeled_encodings = tf.matmul(attention, unlabeled_encodings)
            weighted_labeled_encodings = tf.nn.softmax(weighted_labeled_encodings)

            print(f'{weighted_labeled_encodings.shape=}')
            
            output = self.linear_probe(weighted_labeled_encodings, training=True)
            probe_loss = self.probe_loss(labels, output)
            encoder_gradients = tape.gradient(probe_loss, self.encoder.trainable_weights)
            linear_probe_gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)

            self.pseudo_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_weights))
            self.probe_optimizer.apply_gradients(zip(linear_probe_gradients, self.linear_probe.trainable_weights))

            return output, probe_loss
            '''
            output = self.pseudo_linear_probe(weighted_labeled_encodings)
            pseudo_loss = self.pseudo_loss(labels, output)
            encoder_gradients = tape.gradient(pseudo_loss, self.encoder.trainable_weights)

            self.pseudo_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_weights))
            

        

        # generate probabilities for the labeled data


        # CCE between pseudo-classified data and true labeled

    
        #memory object to store the pseudolabel data
        '''
        Be able to know which points are pseudolabels. Given the unlabeled points, 
        get average distance to the labeled points, softmax and associate 
        a pseudolabel

        '''
    def pad_clusters(self, clusters):
        acc = []
        max_size = max(len(c) for c in clusters)
        # print(f'{max_size=}')

        for c in clusters:
            # print(f'{c.shape=}')
            padding_idx = np.random.choice(np.arange(len(c)), size=(max_size - len(c)), replace=True)
            padding = c[padding_idx]
            acc.append(np.concatenate([c, padding]))
        
        # print(f'{[c.shape for c in acc]}')
        return np.asarray(acc)
    def pseudo_classified_test(self, unlabeled_images, labeled_images, labels):
        unlabeled_encodings = self.encoder(unlabeled_images)
        labeled_encodings = self.encoder(labeled_images)

        # print(f'{type(labels)}')
        # print(f'{tf.unique(labels)=}')
        # tf.cast(labels, dtype=tf.float32)
        # print(f'{labels.shape=}')
        # for c in tf.unique(labels[:,0]):
        #     print(f'{c}')

        # class_indices = [tf.experimental.numpy.nonzero(labels==c)[0] for c in tf.unique(labels[:,0])]
        
        # print(f'{[print(idx.shape) for idx in class_indices]=}\n')

        classes, _ = tf.unique(tf.squeeze(labels))
        # print(f'{classes.shape=}')
        class_indices = [tf.experimental.numpy.nonzero(labels==c)[0] for c in classes]
        # print(f'{class_indices.numpy()=}')
        # print(f'{class_indices.shape=}')
        # print(f'{len(class_indices)=}')
        # print(f"{[c.shape for c in class_indices]}")
        # print(f'{class_indices=}')
        

        # print(f'{labeled_encodings.shape=}')
        # for indices in class_indices:
        #     print(f'{indices=}')
        #     print(f'{type(indices)}')
        labeled_encodings_by_class = [labeled_encodings.numpy()[indices] for indices in class_indices]
        # print(f'{labeled_encodings.shape=}')
        # labeled_encodings_by_class = tf.gather(labeled_encodings, _)
        # print(f'{labeled_encodings_by_class.shape=}')
        # print(f'{len(labeled_encodings_by_class)=}')
        # print(f'{[c.shape for c in labeled_encodings_by_class]=}')
        padded_labeled_encodings_by_class = self.pad_clusters(labeled_encodings_by_class)
        print(f'{unlabeled_encodings.shape=}')
        print(f'{padded_labeled_encodings_by_class.shape=}')
        # unlabeled_labeled_dots_by_class = np.einsum('ij,lji->lij',unlabeled_encodings, padded_labeled_encodings_by_class)
        unlabeled_labeled_dots_by_class = tf.matmul(unlabeled_encodings, padded_labeled_encodings_by_class, transpose_b=True)
        print(f'{unlabeled_labeled_dots_by_class.shape=}')
        averaged = tf.transpose(tf.reduce_mean(unlabeled_labeled_dots_by_class, axis=-1))
        print(f'{averaged.shape=}')
        softmaxed = tf.nn.softmax(averaged)
        pseudo_labeled = tf.argmax(softmaxed, axis=-1)
        print(f'{pseudo_labeled.shape=}')
        # print(f'{len(labeled_encodings_by_class)=}')

        # (num_unlabeled, 128) @ (128, num_in_class_labeled) for c in classes
        # unlabeled_labeled_dots_by_class = [tf.norm(unlabeled_encodings - labeled_encodings_class, ord='euclidean') for labeled_encodings_class in labeled_encodings_by_class]
        # unlabeled_labeled_dots_by_class = [tf.nn.softmax(t, axis=-1) for t in unlabeled_labeled_dots_by_class]
        # print(f'{len(unlabeled_labeled_dots_by_class)=}')
        # attention = tf.convert_to_tensor(unlabeled_labeled_dots_by_class)
        # print(f'{attention.shape=}')
        
        # print('finished!')
        exit()

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

        # TODO: New encoder-specific gradient tape training phase
        # Consider: pseudolabels? 
        
        # class_logits, probe_loss = self.pseudo_classified(unlabeled_images, labeled_images, labels)

        # Below becomes linear-probe specific training phase (i.e. self.encoder(training=False))
        ########################################################################
        # Only labeled images are used WITH LABELS
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )
        '''
        # BEGIN SEMI-SUPERVISED PORTION

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
        
        # With training encoder
        # NOTE: we didn't actually update encoder LMAO
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
                                              split_rate_range=0.5,
                                              contrastive_learning_rate_range=0.01,
                                              probe_learning_rate_range=(0.01,0.03))

    # setting the unique training rates for each part of the model
    gradual_supervised_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_contrastive_learning_rate),
        pseudo_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_contrastive_learning_rate),
        probe_optimizer=keras.optimizers.Adam(gradual_supervised_model.curr_probe_learning_rate),
        # run_eagerly=True # TODO: REMOVE THIS
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
    gradual_supervised_model.save_weights(f'../checkpoints/gradual_pseudo_supervised_model_{title_acc}.weights.h5')
    print('Successfully saved model!')