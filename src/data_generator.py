import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


class DataGenerator(tf.keras.utils.Sequence):
    '''
    Data generator object from a list of samples.
    samples      : list of Pandas DataFrames
    batch_size   : the batch size
    feature_cols : list of feature columns
    target_cols  : list of target columns (even if just one element)
    shuffle      : whether to shuffle the data. Reshuffled after every epoch. 
    returns: a generator giving a list with length batch_size of tuples (sample, target)
    '''
    def __init__(
        self,
        samples,
        batch_size,
        feature_cols,
        target_cols,
        shuffle=True) -> None:
        
        self.samples = samples
        self.batch_size = batch_size
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.shuffle = shuffle
        self.num_targets = len(self.target_cols)
        self.input_shape = self.samples[0].loc[:, self.feature_cols].shape
        self.num_samples = len(self.samples)
        self.indexes = np.arange(self.num_samples)
        self.on_epoch_end()
        

    def __len__(self):
        return int(self.num_samples // self.batch_size)

    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        samples = np.empty((self.batch_size, *self.input_shape))
        targets = np.empty((self.batch_size, self.num_targets))

        # Add batch_size number of samples to a list
        for i, sample_index in enumerate(batch_indexes):
            
            data = self.samples[sample_index]
            
            X = data.loc[:, self.feature_cols].values
            y = data[self.target_cols].values[-1]
            
            samples[i,] = X
            targets[i, :] = y

        return samples, targets

    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)