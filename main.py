from preprocess import *
from dframe_to_dataloader import *
from make_predictions_from_dataloader import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optuna
from LSTMRNNModel import *
from RNNModel import *
from LSTMForecaster import *
from train_valid import *

# ___________________ Pre - Processing ______________________________________________
train_cols = ['cyclic_yday_cos',
              'cyclic_yday_sin',
              'cyclic_sec_cos',
              'cyclic_sec_sin',
              'T_Ext_PV',
              'Cloud',
              'Air0min',
              'Air0max',
              'T_RDC_PV',
              'EV_RDC',
              'Air0min_shift3',
              'Air0max_shift3',
              'Air0min_shift6',
              'Air0max_shift6',
              'Air0min_shift12',
              'Air0max_shift12',
              'Air0min_shift24',
              'Air0max_shift24']

target_cols = 'T_Depart_PV'

cols_2_norm = ['T_Ext_PV', 'Cloud', 'Air0min', 'Air0max', 'T_RDC_PV']

path_small_data = "Export_26-01-2023_au_22-02-2023.csv"
path_full_data = "Export_21-01-2021_au_17-03-2023.csv"
# dframe = preprocess("Export_test_appel.csv")
# dframe, min_out, max_out = preprocess(path_small_data, cols_2_norm, standard_bool=True)
dframe, min_out, max_out = preprocess(path_full_data, cols_2_norm, standard_bool=True)
# dframe = dframe[:round(len(dframe)/10)]

path_small_dframe = "preprocessed_small_frame.csv"
path_full_dframe = "preprocessed_full_frame.csv"

# no need to comment a part on another juste use these bools to choose the operation
we_optimize = False
we_train_and_plot = True


if we_optimize:

    def objective(trial):

        # __________ Dataloaders _____________________
        train_wdw = trial.suggest_int('train_wdw', 1, 4*24)
        target_wdw = trial.suggest_int('target_wdw', 1, 4*12)
        BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 1, 8)
        nb_jours_test = 8
        trainloader, validloader, testloader = dframe_to_dataloader(dframe, train_wdw, target_wdw, train_cols, target_cols,
                                                                    BATCH_SIZE, nb_jours_test)

        # __________ Parameters _____________________
        nhid_lstm = trial.suggest_int('nhid_lstm', 32, 64)  # Number of nodes in the lstm hidden layer
        nhid_rnn = 64  # Number of nodes in the rnn hidden layer
        n_dnn_layers = 1  # Number of hidden fully connected layers
        n_lstm = 1  # Number of lstm layers
        nout = target_wdw  # Prediction Window
        sequence_len = train_wdw  # Training Window
        ninp = len(train_cols)

        lr = 4e-7
        n_epochs = 25
        p_dpout = 0.5
        # ALPHA = 0.001
        # L1_RATIO = 0.5

        # Device selection (CPU | GPU)
        USE_CUDA = torch.cuda.is_available()
        device = 'cuda' if USE_CUDA else 'cpu'

        # Initialize the model
        torch.set_default_dtype(torch.float64)
        model = LSTMForecaster(ninp, nhid_lstm, nout, sequence_len, n_lstm_layers=n_lstm,
                               use_cuda=USE_CUDA, p_dropout=p_dpout).to(device)
        # model = RNNModel(ninp, nhid_rnn, n_dnn_layers, nout, p_dropout=p_dpout).to(device)

        # model = LSTMRNNModel(ninp, nhid_lstm, nhid_rnn, nout, sequence_len, n_lstm_layers=1, n_rnn_layers=1,
        #                      use_cuda=USE_CUDA).to(device)

        # Initialize the loss function and optimizer
        # class Regularization(torch.nn.Module):
        #     def __init__(self, alpha=0.5, l1_ratio=0.5):
        #         super().__init__()
        #         self.alpha = alpha
        #         self.l1_ratio = l1_ratio
        #
        #     def forward(self, model):
        #         l1_norm = torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 1)
        #         l2_norm = torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 2)
        #         return self.alpha * (self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm)

        criterion = torch.nn.MSELoss()
        # elastic_net = Regularization(alpha=ALPHA, l1_ratio=L1_RATIO)
        # criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model, t_losses, v_losses = train_valid(model,
                                                optimizer,
                                                criterion,
                                                n_epochs,
                                                trainloader,
                                                validloader,
                                                device,
                                                trial)

        return v_losses[-1]


    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3,
                                                                   n_warmup_steps=5,
                                                                   interval_steps=2))

    study.optimize(objective, n_trials=20)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

if we_train_and_plot:
    # __________ Dataloaders _____________________
    train_wdw = 77
    target_wdw = 1
    BATCH_SIZE = 3
    nb_jours_test = 8
    trainloader, validloader, testloader = dframe_to_dataloader(dframe, train_wdw, target_wdw, train_cols, target_cols,
                                                                BATCH_SIZE, nb_jours_test)

    # __________ Parameters _____________________
    nhid_lstm = 49  # Number of nodes in the lstm hidden layer
    nhid_rnn = 64  # Number of nodes in the rnn hidden layer
    n_lstm = 3  # Number of lstm layers
    nout = target_wdw  # Prediction Window
    sequence_len = train_wdw  # Training Window
    ninp = len(train_cols)

    lr = 7.88e-07
    n_epochs = 25
    p_dpout = 0.04
    # ALPHA = 0.001
    # L1_RATIO = 0.5

    # Device selection (CPU | GPU)
    USE_CUDA = torch.cuda.is_available()
    device = 'cuda' if USE_CUDA else 'cpu'

    # Initialize the model
    torch.set_default_dtype(torch.float64)
    model = LSTMForecaster(ninp, nhid_lstm, nout, sequence_len, n_lstm_layers=n_lstm,
                           use_cuda=USE_CUDA, p_dropout=p_dpout).to(device)
    # model = RNNModel(ninp, nhid_rnn, n_dnn_layers, nout, p_dropout=p_dpout).to(device)

    # model = LSTMRNNModel(ninp, nhid_lstm, nhid_rnn, nout, sequence_len, n_lstm_layers=1, n_rnn_layers=1,
    #                      use_cuda=USE_CUDA).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trial = 1
    model, t_losses, v_losses = train_valid(model,
                                            optimizer,
                                            criterion,
                                            n_epochs,
                                            trainloader,
                                            validloader,
                                            device,
                                            trial)
    # ___________________ Plotting losses against epochs ______________________________________________
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
        plt.plot(datetime, pred, label="predicted", color='blue')
    plt.plot(datetime, actual, label="data", color='orange')
    plt.legend(loc="upper left")
    plt.ylim(0, 60)

    plt.gcf().autofmt_xdate()

    plt.ylabel("T_Depart_PAC (°C)")
    plt.show()
