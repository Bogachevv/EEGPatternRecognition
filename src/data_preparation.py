import numpy as np
import scipy
import mne

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch_geometric.data import Data, InMemoryDataset

from omegaconf import OmegaConf
from enum import Enum

import pathlib
from tqdm import tqdm

from utils import P300Getter


class GraphMatrixDataset(InMemoryDataset):
    def __init__(self, root, train_raw, graph, data_path, eloc, test_chars=None, filter=True,
                    n_channels=64, sfreq=120, sample_size=72, pos_rate=None, label='model', transform=None, pre_transform=None):

        self.graph = graph
        self.root = root
        self.label = label
        self.data_path = data_path
        self.filter = filter
        self.pos_rate = pos_rate

        self.p300_dataset = P300Getter(train_raw, eloc, n_channels, sfreq, sample_size, target_chars=test_chars)

        super(GraphMatrixDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.root + self.label]

    def download(self):
        pass

    def process(self):
        data_list = []

        edge_index = np.where(self.graph.toarray() == 1)
        edge_index = torch.tensor(np.stack([edge_index[0].flatten(), edge_index[1].flatten()]), dtype=torch.long)

        self.p300_dataset.get_cnn_p300_dataset(filter=self.filter)

        if self.pos_rate:
            self.p300_dataset.upsample(self.pos_rate)

        X_total, y_total = self.p300_dataset.get_data()

        for i in tqdm(range(X_total.shape[0])):
            x = X_total[i]
            y = (torch.arange(0, 2) == y_total[i]).float().unsqueeze(0)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CNNMatrixDataset(Dataset):
    def __init__(self, tensors, with_target=True, transform=None, num_classes=2):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.with_target = with_target
        self.num_classes = num_classes

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        if self.with_target:
            y = self.tensors[1][index]
            if self.num_classes == 2:
                y = y.reshape(-1, ).unsqueeze(1) == torch.arange(0, self.num_classes)
            y = y.float().squeeze()
            return x, y
        else:
            return x

    def __len__(self):
        return self.tensors[0].size(0)

class EEGDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, with_target=True, transform=None, num_classes=2):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.with_target = with_target
        self.num_classes = num_classes

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        if self.with_target:
            y = self.tensors[1][index]
            #y = (torch.arange(0, self.num_classes) == y).float()
            return x, y.unsqueeze(0).float()
        else:
            return x

    def __len__(self):
        return self.tensors[0].size(0)


def load_dataset(config):
    # data_path = pathlib.Path(r'C:\Users\Vladimir\PycharmProjects\EEGPatternRecognition\matrix_dataset')
    data_path = pathlib.Path(config.data_path)

    train_A_raw = scipy.io.loadmat(data_path / 'Subject_A_Train.mat')
    test_A_raw = scipy.io.loadmat(data_path / 'Subject_A_Test.mat')

    eloc = mne.channels.read_custom_montage(data_path / 'eloc64.txt')
    test_A_chars = list('WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU')

    A_train_ds = P300Getter(train_A_raw, eloc, sample_size=72)
    A_test_ds = P300Getter(test_A_raw, eloc, sample_size=72, target_chars=test_A_chars)

    A_train_ds.get_cnn_p300_dataset(filter=True)
    A_test_ds.get_cnn_p300_dataset(filter=True)

    A_train_ds.upsample(4)

    X_train, y_train = A_train_ds.get_data()
    X_test, y_test = A_test_ds.get_data()

    train_dataset = CNNMatrixDataset(tensors=(X_train, y_train), with_target=True, transform=None)
    test_dataset = CNNMatrixDataset(tensors=(X_test, y_test), with_target=True, transform=None)

    return train_dataset, test_dataset


def get_dataloaders(config, train_dataset, test_dataset):
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.train_bs, 
        shuffle=True
    )

    test_dataloader =  DataLoader(
        dataset=test_dataset, 
        batch_size=config.test_bs, 
        shuffle=False
    )

    return train_dataloader, test_dataloader
