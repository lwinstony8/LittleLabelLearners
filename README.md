# Little Label Learners: Gradual Supervision Approach for Novel Label Learning
## CSCI2470: Deep Learning Final Project

## Developers
Winston Li, JR Byers, Marcel Mateos-Salles

## Setup commands
This project was completed and ran on Brown University's Oscar supercomputer. Primarily, we use
`TensorFlow 2` and `Keras`.

To run locally, we recommend creating a new environment in the root directory of the repository:

`python venv -m /lll.venv`

Next, install the required packages by using the follow command:

`pip install -r requiements.txt`

Alternatively if you have access to Oscar CCV, we recommend following 
[this tutorial](https://docs.ccv.brown.edu/oscar/gpu-computing/installing-frameworks-pytorch-tensorflow-jax/installing-tensorflow)

This will allow you to create an apptainer containing `Tensorflow` and other dependencies. Note, you may have to upgrade `Keras` as follows: `pip install keras --upgrade --user`

## Running commands
To run the models, first `cd` into the `model/` directory: `cd /model`. Then, you can run `python baseline_model.py`, `python gradual_supervised_model.py`, `python self_supervised_model.py`, or `python encoder_training_sandbox.py`.

If running for the first time, all models have a function call `download_data`, which will use `tf.load()` to download the `CIFAR-10` dataset. This is a lightweight dataset comprising 60000 32x32 RGB images at less than 200MB. This will only ever be called once; i.e. if you run `python baseline_model.py` and then `python self_supervised_model.py`, only one set of data will be downloaded.

## Model overview
All models have an `Encoder` layer which features a series of `Conv2D` layers for extracting image features and a `Dense` layer for embedding the features. They also have a `Linear Probe` layer, which projects the embeddings into classification space.

### Baseline Model
`/model/baseline_model.py`

This model only has an `Encoder` followed by a `Linear Probe`. It is fully supervised and thus only uses labeled data.

### Semi-supervised Model
`/model/semi_supervised_model.py`

This is a basic implementation of the *SimCLR* architecture. Beginning with an `Encoder`, embeddings can be projected via a `Projection Head` for contrastive training or be projected via the `Linear Probe` for classification. Note, as this is a purely self-supervised method, *labels* **do not** update the `Encoder`'s weights. Instead, *labeled* and *unlabeled* images are bundled together to train the `Encoder` without any labels, but only the *labeled images* and corresponding *labels* are used for training the `Linear Probe`.

In our case, we choose to train the `Linear Probe` while the `Encoder` is training to get real-time feedback of both layers. One could also choose to just train the `Linear Probe` after the `Encoder` has finished training.

### Supervised-Finetuning Model
`/model/supervised_finetuning_model.py`

This model uses a trained `Encoder` and a `Linear Probe`. *Labeled* images and corresponding *labels* are used to update **both** layers at once, hence supervised fine-tuning.

This model was implemented using an older procedure and is thus no longer supported.

### Gradual-Supervised Model
`/model/gradual_supervised_model.py`

This model is similar to the `semi_supervised_model.py`. However, during the `Linear Probe` training step, it uses the classification gradients to **also** update the `Encoder`. Unlike `supervised_finetuning_model.py`, this is a semi-supervised method as not all data are labeled, nor are all classes represented at each epoch.

### Encoder Training Sandbox
`/model/encoder_training_sandbox.py`

This model is similar to `gradual_supervised_model.py`, but it features additional `Encoder` training steps in-between contrastive training and linear probe classification training. Namely, there are several *pseudo-* and *meta-labeling* methods given.

Documentation is written in the file for detailed explanations/motivations behind these methods.

### Custom Callbacks
`/model/custom_callbacks.py`

This contains the `ScheduledSubsetCallback` class, which contains several methods responsible for providing the parameters for splitting up the data, as used by `Dataloader`.

First, it handles class representation such that at later epochs, images from more classes are represented.

Second, it handles label/unlabeled data split rate such that at later epochs, the model is given more labeled samples to work with.

Third, it handles learning rate updates such that at later epochs, the weight updates are smaller at an exponential decay rate.

### Dataloader
`/data/dataloader.py`

This contains the `Dataloader` class, which stores data information and methods for splitting up the data. It is intended to be used as an instance variable of a `model`.

In general, it saves train/test features and labels. It also contains methods for subsetting the data to contain only few labels (i.e. few classes), as well as randomly choosing indices at a given `split_rate` to determine how many samples should be labeled or not. Finally, it handles converting the `numpy` data into `TensorFlow` data batches.