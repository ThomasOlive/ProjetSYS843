from preprocess import *
from dframe_to_dataloader import *
from make_predictions_from_dataloader import *
from LSTMRNNModel import *
from RNNModel import *
from LSTMForecaster import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ___________________ Pre - Processing ______________________________________________
train_cols = ['cyclic_yday_cos', 'cyclic_yday_sin', 'cyclic_sec_cos', 'cyclic_sec_sin', 'T_Ext_PV', 'Cloud', 'Air0min',
              'Air0max', 'T_RDC_PV', 'EV_RDC', 'EV_3Voies', 'EV_1ER']
target_cols = 'T_Depart_PV'

cols_2_norm = ['T_Ext_PV', 'Cloud', 'Air0min', 'Air0max', 'T_RDC_PV']

path_small_data = "Export_26-01-2023_au_22-02-2023.csv"
path_full_data = "Export_21-01-2021_au_17-03-2023.csv"
# dframe = preprocess("Export_test_appel.csv")
# dframe, min_out, max_out = preprocess(path_small_data, cols_2_norm, standard_bool=True)
dframe, min_out, max_out = preprocess(path_full_data, cols_2_norm, standard_bool=True)
dframe = dframe[:round(len(dframe)/3)]
path_small_dframe = "preprocessed_small_frame.csv"
path_full_dframe = "preprocessed_full_frame.csv"
# dframe = pd.read_csv(path_small_dframe)

train_wdw = 24
target_wdw = 24
BATCH_SIZE = 6*4
nb_jours_test = 8
trainloader, validloader, testloader = dframe_to_dataloader(dframe, train_wdw, target_wdw, train_cols, target_cols,
                                                            BATCH_SIZE, nb_jours_test)


# ___________________ Training the model ______________________________________________

# __________ Parameters _____________________
nhid_lstm = 64  # Number of nodes in the lstm hidden layer
nhid_rnn = 64  # Number of nodes in the rnn hidden layer
n_dnn_layers = 1  # Number of hidden fully connected layers
n_lstm = 1  # Number of lstm layers
nout = target_wdw  # Prediction Window
sequence_len = train_wdw  # Training Window

# Number of input features
ninp = len(train_cols)

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'


# Set learning rate and number of epochs to train over
lr = 4e-7
n_epochs = 110
p_dpout = 0.5

# Initialize the model
torch.set_default_dtype(torch.float64)
model = LSTMForecaster(ninp, nhid_lstm, nout, sequence_len, n_deep_layers=n_dnn_layers, n_lstm_layers=n_lstm,
                       use_cuda=USE_CUDA, p_dropout=p_dpout).to(device)
# model = RNNModel(ninp, nhid_rnn, n_dnn_layers, nout, p_dropout=p_dpout).to(device)

# model = LSTMRNNModel(ninp, nhid_lstm, nhid_rnn, nout, sequence_len, n_lstm_layers=1, n_rnn_layers=1,
#                      use_cuda=USE_CUDA).to(device)


# Initialize the loss function and optimizer
class ElasticNetRegularization(torch.nn.Module):
    def __init__(self, alpha=0.5, l1_ratio=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, model):
        l1_norm = torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 1)
        l2_norm = torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 2)
        return self.alpha * (self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm)


criterion = torch.nn.MSELoss()
# elastic_net = ElasticNetRegularization(alpha=0.0001, l1_ratio=0.75)
# criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# _________ Training ___________________________
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
        # loss = criterion(preds, y) + elastic_net(model)
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
            preds = model(x).squeeze()
            error = criterion(preds, y)
            # error = criterion(preds, y) + elastic_net(model)
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

pred, actual, k_idx = make_predictions_from_dataloader(model, testloader, dframe, device, target_wdw)

for idx, x in enumerate(pred):
    pred[idx] = x * (max_out - min_out) + min_out

for idx, x in enumerate(actual):
    actual[idx] = x * (max_out - min_out) + min_out

datetime = pd.to_datetime(dframe['date'])
datetime = datetime[k_idx:k_idx+len(pred)]
if target_wdw > 1:
    for idx, row in enumerate(pred):
        date = datetime[idx:idx+len(row)]
        if len(date) == len(row):
            plt.plot(date, row)
else:
    plt.plot(datetime, pred, label="predicted")
plt.plot(datetime, actual, label="data", color='orange')
plt.legend(loc="upper left")
plt.ylim(0, 60)

plt.gcf().autofmt_xdate()

plt.ylabel("T_Depart_PAC (°C)")
plt.show()
