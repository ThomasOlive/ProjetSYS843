import torch
import torch.nn as nn
import numpy as np

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'


class LSTMRNNModel(nn.Module):

    def __init__(self, n_features, hidden_dim_lstm, hidden_dim_rnn, n_outputs, sequence_len, n_lstm_layers=1,
                 n_rnn_layers=1, n_deep_layers=10, use_cuda=False):

        # n_features: number of input features
        # n_hidden: number of neurons in each hidden layer
        # n_outputs: number of outputs to predict for each training example
        # n_deep_layers: number of hidden dense layers after the lstm layer
        # sequence_len: number of steps to look back at for prediction

        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.hidden_dim_lstm = hidden_dim_lstm
        self.use_cuda = use_cuda  # set option for device selection

        # LSTM Layer
        self.lstm = nn.LSTM(n_features,
                            hidden_dim_lstm,
                            num_layers=n_lstm_layers,
                            batch_first=True)  # As we have transformed our data in this way

        # first dense after lstm
        self.fc_lstm = nn.Linear(hidden_dim_lstm * sequence_len, n_outputs)


        # Defining the number of layers and the nodes in each layer
        self.hidden_dim_rnn = hidden_dim_rnn
        self.n_rnn_layers = n_rnn_layers

        # RNN layers
        self.rnn = nn.RNN(n_features, hidden_dim_rnn, n_rnn_layers, batch_first=True)
        # Fully connected layer
        self.fc_rnn = nn.Linear(hidden_dim_rnn, n_outputs)

        self.fc_last = nn.Linear(n_outputs*2, n_outputs)

    def forward(self, x):

        # __________ LSTM _________________
        # Initialize hidden state
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.hidden_dim_lstm)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.hidden_dim_lstm)

        # move hidden state to device
        if self.use_cuda:
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)

        self.hidden = (hidden_state, cell_state)

        # Forward Pass
        out_lstm, h = self.lstm(x, self.hidden)  # LSTM # Flatten lstm out
        out_lstm = self.fc_lstm(out_lstm.contiguous().view(x.shape[0], -1))  # First Dense

        # __________ RNN _________________
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.n_rnn_layers, x.size(0), self.hidden_dim_rnn).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out_rnn, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out_rnn = out_rnn[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out_rnn = self.fc_rnn(out_rnn)

        # __________ Hybrid _________________
        out = [out_lstm, out_rnn]
        out = torch.Tensor(out)
        out = self.fc_last(out)
        return out  # Pass forward through fully connected DNN.
