import keras
import time
import matplotlib.pyplot as plt

class ScheduledSubsetCallback_Old(keras.callbacks.Callback):
    def __init__(self, cur_epoch):
        super().__init__()
        self.cur_epoch = cur_epoch
        # print(f'reached callback epoch {self.cur_epoch}')
        
    def on_epoch_begin(self, epoch, logs=None):

        # NOTE: These if statements allow us to short-cut out of re-generating datasets
        if self.cur_epoch > self.model.dataloader.num_classes:
            return
        subset_size = self.cur_epoch
        if subset_size < self.model.floor_num_classes:
            return
        if subset_size > self.model.ceiling_num_classes+1:
            # print('Reached ceiling!')
            # print(f'{subset_size=}')
            # print(f'{self.model.ceiling_num_classes=}')
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
class TimingCallback(keras.callbacks.Callback):
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
    def __init__(self, model):
        self.model = model

    def __call__(self, cur_epoch):
        # NOTE: These if statements allow us to short-cut out of re-generating datasets
        if cur_epoch > self.model.dataloader.num_classes:
            return
        subset_size = cur_epoch # TODO: this can be a much more complicated update method...
        if subset_size <= self.model.floor_num_classes:
            return
        if subset_size > self.model.ceiling_num_classes:
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