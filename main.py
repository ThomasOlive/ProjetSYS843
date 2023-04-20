import pandas as pd
from datetime import datetime, timedelta
import pytz
from preprocess import *
from dframe_to_dataloader import *
from LSTMForecaster import *
from make_predictions_from_dataloader import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset, random_split
import torch.nn as nn
import numpy as np
from itertools import chain

# ___________________ Pre - Processing ______________________________________________
train_cols = ['cyclic_yday_cos', 'cyclic_yday_sin', 'cyclic_sec_cos', 'cyclic_sec_sin', 'T_Ext_PV', 'Cloud', 'Air0min',
              'Air0max', 'T_RDC_PV', 'EV_RDC']
cols_2_norm = ['T_Ext_PV', 'Cloud', 'Air0min', 'Air0max', 'T_RDC_PV', 'EV_RDC']

path_small_data = "Export_26-01-2023_au_22-02-2023.csv"
path_full_data = "Export_21-01-2021_au_17-03-2023.csv"
# dframe = preprocess("Export_test_appel.csv")
dframe = preprocess(path_small_data, cols_2_norm, standard_bool=True)
# dframe = preprocess(path_full_data, cols_2_norm, standard_bool=True)

# path_full_dframe = "drive/MyDrive/Colab Notebooks/SYS843/preprocessed_full_frame.csv"
# dframe = pd.read_csv(path_full_dframe)

target_cols = 'T_Depart_PV'
train_wdw = int(2*4)
target_wdw = 1
BATCH_SIZE = 1
nb_jours_test = 4
trainloader, validloader, testloader = dframe_to_dataloader(dframe, train_wdw, target_wdw, train_cols, target_cols,
                                                            BATCH_SIZE, nb_jours_test)


# ___________________ Training the model ______________________________________________
nhid = 8  # Number of nodes in the hidden layer
n_dnn_layers = 1  # Number of hidden fully connected layers
n_lstm = 1  # Number of lstm layers
nout = target_wdw  # Prediction Window
sequence_len = train_wdw  # Training Window

# Number of input features
ninp = len(train_cols)

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

# Initialize the model
torch.set_default_dtype(torch.float64)
model = LSTMForecaster(ninp, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers, n_lstm_layers=n_lstm, use_cuda=USE_CUDA).to(device)
# model = model.float()
# Set learning rate and number of epochs to train over
lr = 4e-4
n_epochs = 2

# Initialize the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


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

        # Forward Pass
        preds = model(x).squeeze()
        loss = criterion(preds, y)  # compute batch loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_loss / len(trainloader)
    t_losses.append(epoch_loss)

    # validation step
    model.eval()
    # Loop over validation dataset
    for x, y in validloader:
        with torch.no_grad():
            x = x.to(device)
            y = y.squeeze().to(device)
            # x, y = x.to(device), y.to(device)
            preds = model(x).squeeze()
            # preds = model(x)
            error = criterion(preds, y)
        valid_loss += error.item()
    valid_loss = valid_loss / len(testloader)
    v_losses.append(valid_loss)

    print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
# plot_losses(t_losses, v_losses)

plt.plot(range(n_epochs), t_losses, label="train")
plt.plot(range(n_epochs), v_losses, label="valid")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("MSE loss")
plt.show()


# ___________________ Testing the model ______________________________________________
device = 'cpu'
model = model.to(device)

pred, actual, k_idx = make_predictions_from_dataloader(model, testloader, dframe, device)

datetime = pd.to_datetime(dframe['date'])
datetime = datetime[k_idx:k_idx+len(pred)]

plt.plot(datetime, pred, label="predicted")
plt.plot(datetime, actual, label="data")
plt.legend(loc="upper left")
plt.ylim(0, 60)

plt.gcf().autofmt_xdate()

plt.ylabel("T_Depart_PAC (°C)")
plt.show()
