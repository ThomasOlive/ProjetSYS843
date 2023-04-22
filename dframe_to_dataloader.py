import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch


# Defining a function that creates sequences and targets
def generate_sequences(df: pd.DataFrame, tw: int, pw: int, training_columns, target_columns):
    # df: Pandas DataFrame of the time-series
    # tw: Training Window - Integer defining how many steps to look back
    # pw: Prediction Window - Integer defining how many steps forward to predict
    # returns: dictionary of sequences and targets for all sequences

    data = dict()  # Store results into a dictionary
    L = len(df)
    idx = 0
    for i in range(L - tw):
        # presence = True
        # for Air0max in df.iloc[i:i+tw]['Air0max']:
        #   if Air0max != check_presence:
        #     presence = False
        # if presence:
        # Get current sequence
        sequence = df.iloc[i:i + tw][training_columns].values
        # Get values right after the current sequence
        target = df.iloc[i + tw:i + tw + pw][target_columns].values

        data[idx] = {'sequence': sequence, 'target': target}
        idx = idx + 1
    return data


def generate_dict_batch(dict_seq: dict, batch_sz: int, tw: int, pw: int, training_columns):
    idx = 0
    l_dict_seq = len(dict_seq)
    nb_batch = l_dict_seq // batch_sz

    batch_x = np.zeros([nb_batch, batch_sz, tw, len(training_columns)])
    batch_y = np.zeros([nb_batch, batch_sz, pw, 1])

    batch_idx = 0
    in_batch_idx = 0
    for dict_idx in range(l_dict_seq):
        if batch_idx < nb_batch:

            batch_x[batch_idx][in_batch_idx][:] = dict_seq[dict_idx]['sequence']
            if pw == 1:
                batch_y[batch_idx][in_batch_idx][:] = dict_seq[dict_idx]['target']
            else:
                batch_y[batch_idx][in_batch_idx][:] = dict_seq[dict_idx]['target'][:, np.newaxis]

            in_batch_idx += 1
            if in_batch_idx == batch_sz:
                in_batch_idx = 0
                batch_idx += 1
    return batch_x, batch_y


class SequenceDataset(Dataset):

    def __init__(self, df_x, df_y):
        self.data_x = df_x
        self.data_y = df_y

    def __getitem__(self, idx):
        sample_x = self.data_x[idx]
        sample_y = self.data_y[idx]
        x = torch.from_numpy(sample_x).type(torch.float64)
        y = torch.from_numpy(sample_y).type(torch.float64)
        returned = x, y
        # returned = torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])
        return returned

    def __len__(self):
        return len(self.data_x)


def dframe_to_dataloader(dframe, train_wdw, target_wdw, train_cols, target_cols, BATCH_SIZE=1, nb_jours_test=4):

    dict_seq = generate_sequences(dframe, train_wdw, target_wdw, train_cols, target_cols)

    # removing last targets because they don't have the correct target window size
    len_dict = len(dict_seq)
    for i in range(len_dict - target_wdw + 1, len_dict):
        dict_seq.pop(i)

    batch_x, batch_y = generate_dict_batch(dict_seq, BATCH_SIZE, train_wdw, target_wdw, train_cols)

    dataset = SequenceDataset(batch_x, batch_y)

    test_split = round(len(dataset) - 24 * 4 * nb_jours_test / BATCH_SIZE)  # nb_jours_test derniers jours
    test_ds = Subset(dataset, range(test_split, len(dataset)))

    train_split = round(0.8 * test_split)

    train_ds = Subset(dataset, range(0, train_split))
    valid_ds = Subset(dataset, range(train_split, test_split))

    # random_subset = Subset(dataset, range(0, test_split))
    # train_ds, valid_ds = random_split(random_subset, [0.8, 0.2])

    trainloader = DataLoader(train_ds, batch_size=None, batch_sampler=None, shuffle=True)
    validloader = DataLoader(valid_ds, batch_size=None, batch_sampler=None)
    testloader = DataLoader(test_ds, batch_size=None, batch_sampler=None)
    return trainloader, validloader, testloader

