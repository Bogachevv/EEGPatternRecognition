import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

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
