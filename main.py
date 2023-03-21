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
from LSTMForecaster import *


train_cols = ['yday', 'total_sec', 'T_Ext_PV', 'Cloud', 'Air0min', 'Air0max', 'T_RDC_PV', 'EV_RDC']
# dframe = preprocess("Export_test_appel.csv")
# dframe = preprocess("Export_21-01-2021_au_17-03-2023.csv")
dframe = preprocess("Export_26-01-2023_au_22-02-2023.csv", train_cols)


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


train_cols = ['yday', 'total_sec', 'T_Ext_PV', 'Cloud', 'Air0min', 'Air0max', 'T_RDC_PV', 'EV_RDC']
target_cols = 'T_Depart_PV'

train_wdw = 4*48
target_wdw = 4
dict_seq = generate_sequences(dframe, train_wdw, target_wdw, train_cols, target_cols)

# removing last targets because they don't have the correct window size
len_dict = len(dict_seq)
for i in range(len_dict - target_wdw+1, len_dict):
    dict_seq.pop(i)


# Here we are defining properties for our model

BATCH_SIZE = 1  # Training batch size
split = 0.7  # Train/Test Split ratio

dataset = SequenceDataset(dict_seq)

# Split the data according to our split ratio and load each subset into a
# separate DataLoader object
train_len = int(len(dataset)*split)
lens = [train_len, len(dataset)-train_len]
lens = [len(dataset)*0.7, len(dataset)*0.2, len(dataset)*0.1]
lens = [0.7, 0.2, 0.1]
train_ds, test_ds, mytest_ds = random_split(dataset, lens)
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
mytestloader = DataLoader(mytest_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


nhid = 10  # Number of nodes in the hidden layer
n_dnn_layers = 1  # Number of hidden fully connected layers
n_lstm = 1  # Number of hidden fully connected layers
nout = target_wdw  # Prediction Window
sequence_len = train_wdw  # Training Window

# Number of features (since this is a univariate timeseries we'll set
# this to 1 -- multivariate analysis is coming in the future)
ninp = len(train_cols)

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

# Initialize the model
model = LSTMForecaster(ninp, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers, n_lstm_layers=n_lstm, use_cuda=USE_CUDA).to(device)

# Set learning rate and number of epochs to train over
lr = 4e-4
n_epochs = 2

# Initialize the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# Lists to store training and validation losses
t_losses, v_losses = [], []
# Loop over epochs
for epoch in range(n_epochs):
    train_loss, valid_loss = 0.0, 0.0

    # train step
    model.train()
    # Loop over train dataset
    for x, y in trainloader:
        optimizer.zero_grad()
        # move inputs to device
        x = x.to(device)
        y = y.squeeze().to(device)
        # y = y.to(device)
        # Forward Pass
        preds = model(x).squeeze()
        # preds = model(x)
        loss = criterion(preds, y)  # compute batch loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_loss / len(trainloader)
    t_losses.append(epoch_loss)

    # validation step
    model.eval()
    # Loop over validation dataset
    for x, y in testloader:
        with torch.no_grad():
            x, y = x.to(device), y.squeeze().to(device)
            # x, y = x.to(device), y.to(device)
            preds = model(x).squeeze()
            # preds = model(x)
            error = criterion(preds, y)
        valid_loss += error.item()
    valid_loss = valid_loss / len(testloader)
    v_losses.append(valid_loss)

    print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
# plot_losses(t_losses, v_losses)

plt.plot(range(n_epochs), t_losses)
plt.plot(range(n_epochs), v_losses)
plt.show()
#
torch.save(model.state_dict(), 'saved_model.pt')
#
# # Initialize again the model
# checkpoint = torch.load('state_dict.pt')
model = LSTMForecaster(ninp, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers, n_lstm_layers=n_lstm, use_cuda=USE_CUDA).to(device)
model.load_state_dict(torch.load('saved_model.pt'))
# model.eval()
#


def make_predictions_from_dataloader(model, unshuffled_dataloader):
    model.eval()
    x_array, predictions, actuals = [], [], []
    for x, y in unshuffled_dataloader:
        with torch.no_grad():
            # x_array = np.append(x_array, x)
            p = model(x)
            predictions = np.append(predictions, p)
            actuals = np.append(actuals, y.squeeze())
        # print(predictions)
        # predictions = torch.cat(predictions).numpy()
        # actuals = torch.cat(actuals).numpy()
    return x_array, predictions.squeeze(), actuals


x_arr, pred, actual = make_predictions_from_dataloader(model, mytestloader)


plt.plot(range(len(pred)), pred)
plt.plot(range(len(pred)), actual)
plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
