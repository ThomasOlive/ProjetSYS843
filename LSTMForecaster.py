import torch
import torch.nn as nn

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'


class LSTMForecaster(nn.Module):

    def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False):

        # n_features: number of input features
        # n_hidden: number of neurons in hidden layer
        # n_outputs: number of outputs to predict for each training example
        # n_deep_layers: number of hidden dense layers after the lstm layer
        # sequence_len: number of steps to look back at for prediction

        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.nhid = n_hidden
        self.use_cuda = use_cuda  # set option for device selection

        # LSTM Layer
        self.lstm = nn.LSTM(n_features,
                            n_hidden,
                            num_layers=n_lstm_layers,
                            batch_first=True)  # As we have transformed our data in this way

        # first dense after lstm
        self.fc = nn.Linear(n_hidden * sequence_len, n_outputs)

    def forward(self, x):

        # Initialize hidden state
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)

        # move hidden state to device
        if self.use_cuda:
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)

        self.hidden = (hidden_state, cell_state)

        # Forward Pass
        x, h = self.lstm(x, self.hidden)  # LSTM # Flatten lstm out
        x = self.fc(x.contiguous().view(x.shape[0], -1))  # First Dense
        return x  # Pass forward through fully connected DNN.
