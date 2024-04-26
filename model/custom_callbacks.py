import keras

class ScheduledSubsetCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        print('reached callback')
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return
        subset_size = min(epoch, self.model.dataloader.num_classes)
        subset_size = max(subset_size, self.model.base_num_classes) # begin with 5 classes
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