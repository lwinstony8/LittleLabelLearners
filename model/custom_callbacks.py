import keras
import time
import matplotlib.pyplot as plt
import hyperparameters as hp

class ScheduledSubsetCallback_old(keras.callbacks.Callback):
    def __init__(self, cur_epoch):
        super().__init__()
        self.cur_epoch = cur_epoch
        # print(f'reached callback epoch {self.cur_epoch}')
        
    def on_epoch_begin(self, epoch, logs=None):

        # NOTE: These if statements allow us to short-cut out of re-generating datasets
        if self.cur_epoch > self.model.dataloader.num_classes:
            return
        subset_size = self.cur_epoch
        if subset_size < self.model.num_classes_range[0] or subset_size >= self.model.num_classes_range[1]:
            return
        if self.cur_epoch == 0:
            return
        
        '''
        ## Explicitly re-generate subsets
        # subset_size = min(self.cur_epoch, self.model.dataloader.num_classes)
        # subset_size = max(subset_size, self.model.floor_num_classes) # lower bound floor_num_classes
        # subset_size = min(subset_size, self.model.ceiling_num_classes) # upper bound ceiling_num_classes
        '''
        self.model.dataloader.generate_subsets(subset_size=subset_size)
        ## NOTE: this might be put inside ScheduledSubsetCallback
        # this will get replaced once we have the other scheduler
        self.model.dataloader.prepare_dataset(
            self.model.dataloader.x_train_subset, 
            self.model.dataloader.y_train_subset, 
            self.model.dataloader.x_test, 
            self.model.dataloader.y_test)
        
        print(f'{subset_size=}')
        print(f'{self.model.dataloader.x_train_subset.shape=}')

# TODO: Convert to fit big-epoch-loop instead
class TimingCallback_old(keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
    def on_epoch_begin(self, epoch, logs=None):
        self.timetaken = time.perf_counter()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch,time.perf_counter() - self.timetaken))
    def on_train_end(self,logs = {}):
        plt.xlabel('Epoch')
        plt.ylabel('Time per Epoch')
        plt.plot(*zip(*self.times))
        plt.show()
        plt.savefig('times per epoch')
        
class ScheduledSubsetCallback():
    def __init__(self, model: keras.Model):
        """ Initializer for ScheduledSubsetCallback object. Takes in a Model object (i.e. BaselineModel, ContrastiveModel, etc.)

        Args:
            model (keras.Model)
        """        
        self.model = model

    def __call__(self, cur_epoch: int):
        """ Uses cur_epoch to determine rate of subsetting, label splitting, and learning rate
            Will check to make sure subset_size, split_rate, and learning_rate are within given model's respective ranges

        Args:
            cur_epoch (int)
        """        
        subset_size = cur_epoch # TODO: this can be a much more complicated update method...
        subset_size = max(subset_size, self.model.num_classes_range[0])        
        subset_size = min(subset_size, self.model.num_classes_range[1])   

        # Ensure split_rate is within range
        split_rate = cur_epoch / hp.num_epochs
        split_rate = max(split_rate, self.model.split_rate_range[0])
        split_rate = min(split_rate, self.model.split_rate_range[1])

        # Ensure learning_rate is within range
        # this formula could easily change to something more complex, in this case starting out higher
        learning_rate = 0.01 - (0.001*cur_epoch)
        learning_rate = max(learning_rate, self.model.learning_rate_range[0])
        learning_rate = min(learning_rate, self.model.learning_rate_range[1])

        # If subset_size AND split_rate AND learning rate have not changed
        if subset_size == self.model.cur_num_classes and split_rate == self.model.cur_split_rate and learning_rate == self.model.curr_learning_rate:
            return
        
        # Update cur_num_classes/split_rate/curr_learning_rate if changed!
        self.model.cur_num_classes = subset_size
        self.model.cur_split_rate = split_rate
        self.model.curr_learning_rate = learning_rate
                
        self.model.dataloader.generate_subsets(subset_size=subset_size)
        ## NOTE: this might be put inside ScheduledSubsetCallback
        # this will get replaced once we have the other scheduler
        self.model.dataloader.prepare_dataset(
            self.model.dataloader.x_train_subset, 
            self.model.dataloader.y_train_subset, 
            self.model.dataloader.x_test, 
            self.model.dataloader.y_test,
            split_rate=split_rate)
                
        print(f'\n{subset_size=}; {self.model.dataloader.x_train_subset.shape=}; {split_rate=}; {learning_rate=}')
