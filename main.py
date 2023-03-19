# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from datetime import datetime, timedelta
import pytz
from preprocess import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torch.nn as nn
import numpy as np


# dframe = preprocess("Export_test_appel.csv")
# dframe = preprocess("Export_21-01-2021_au_17-03-2023.csv")
dframe = preprocess("Export_26-01-2023_au_22-02-2023.csv")


# print(dframe)

# dframe.plot(x='date', y='T_RDC_PV')
# dframe.plot(x='date', y='T_Ext_PV')
# plt.show()
# rows = range(1, 3)
# col = ['date', 'yday']
# print(dframe.iloc[rows][col].values)


# Defining a function that creates sequences and targets
def generate_sequences(df: pd.DataFrame, tw: int, pw: int, training_columns, target_columns, drop_targets=False):

    # df: Pandas DataFrame of the univariate time-series
    # tw: Training Window - Integer defining how many steps to look back
    # pw: Prediction Window - Integer defining how many steps forward to predict
    # returns: dictionary of sequences and targets for all sequences

    data = dict()  # Store results into a dictionary
    L = len(df)
    for i in range(L-tw):
        # Option to drop target from dataframe
        if drop_targets:
          df.drop(target_columns, axis=1, inplace=True)

        # Get current sequence
        sequence = df.iloc[i:i+tw][training_columns].values
        # Get values right after the current sequence
        target = df.iloc[i+tw:i+tw+pw][target_columns].values

        data[i] = {'sequence': sequence, 'target': target}
    return data




class SequenceDataset(Dataset):

    def __init__(self, df):
        self.data = df

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])

    def __len__(self):
        return len(self.data)


train_cols = ['yday', 'total_sec', 'T_Ext_PV', 'Cloud', 'Air0min', 'Air0max', 'T_RDC_PV']
target_cols = 'T_Depart_PV'
dict_seq = generate_sequences(dframe, 4, 1, train_cols, target_cols)

# Here we are defining properties for our model

BATCH_SIZE = 16  # Training batch size
split = 0.8  # Train/Test Split ratio

dataset = SequenceDataset(dict_seq)


# Split the data according to our split ratio and load each subset into a
# separate DataLoader object
train_len = int(len(dataset)*split)
lens = [train_len, len(dataset)-train_len]
train_ds, test_ds = random_split(dataset, lens)
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
