import numpy as np
import scipy

from omegaconf import OmegaConf
from enum import Enum

import pathlib
import pickle

import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage, DigMontage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import Scaler


class P300Getter:
    def __init__(self, raw_data, eloc, n_channels=64, sfreq=120, sample_size=72, target_chars=None):
        self.raw_data = raw_data
        self.target_chars = target_chars
        self.sample_size = sample_size
        self.n_channels = n_channels

        self._X = None
        self._y = None

        self.eloc = eloc
        self.info = mne.create_info(ch_names=eloc.ch_names, ch_types=['eeg'] * n_channels, sfreq=sfreq)
        self.scaler = Scaler(self.info)

        self.row_dict = []
        self.row_dict += list(zip(['A', 'B', 'C', 'D', 'E', 'F'], [7] * 6))
        self.row_dict += list(zip(['G', 'H', 'I', 'J', 'K', 'L'], [8] * 6))
        self.row_dict += list(zip(['M', 'N', 'O', 'P', 'Q', 'R'], [9] * 6))
        self.row_dict += list(zip(['S', 'T', 'U', 'V', 'W', 'X'], [10] * 6))
        self.row_dict += list(zip(['Y', 'Z', '1', '2', '3', '4'], [11] * 6))
        self.row_dict += list(zip(['5', '6', '7', '8', '9', '_'], [12] * 6))
        self.row_dict = dict(self.row_dict)

        self.col_dict = []
        self.col_dict += list(zip(['A', 'G', 'M', 'S', 'Y', '5'], [1] * 6))
        self.col_dict += list(zip(['B', 'H', 'N', 'T', 'Z', '6'], [2] * 6))
        self.col_dict += list(zip(['C', 'I', 'O', 'U', '1', '7'], [3] * 6))
        self.col_dict += list(zip(['D', 'J', 'P', 'V', '2', '8'], [4] * 6))
        self.col_dict += list(zip(['E', 'K', 'Q', 'W', '3', '9'], [5] * 6))
        self.col_dict += list(zip(['F', 'L', 'R', 'X', '4', '_'], [6] * 6))
        self.col_dict = dict(self.col_dict)
    
    def filter(self, X, freq=120):
        train_array = mne.io.RawArray(X.T, self.info, verbose=False)
        train_array.set_montage(self.eloc)
        self.info = mne.create_info(ch_names=self.eloc.ch_names, ch_types=['eeg'] * self.n_channels, sfreq=freq)
        return train_array.filter(1, 20, method='fir', verbose=False).resample(freq, verbose=False).get_data()

    def to_tensor(self, X_train, y_train):
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)

        return X_train.float(), y_train.float()

    def upsample(self, pos_rate=4):
        pos_idx = np.where(self._y == 1)[0]
        pos_X = self._X[pos_idx]
        pos_y = np.ones(len(pos_idx))

        for _ in range(pos_rate):
            self._X = np.concatenate([self._X, pos_X])
            self._y = np.concatenate([self._y, pos_y])

    def get_cnn_p300_dataset(self, filter=False):
        X = []
        y = []

        for epoch_num in tqdm(range(len(self.raw_data['Flashing']))):
            epoch_flash = self.raw_data['Flashing'][epoch_num]
            idx = np.where(epoch_flash[:-1] != epoch_flash[1:])[0][1::2] + 1
            idx = np.concatenate([[0], idx])
            
            if filter:
                data = self.filter(self.raw_data['Signal'][epoch_num])
            else:
                data = self.raw_data['Signal'][epoch_num].T

            res = []
            bias = 24
            for i in idx:
                res.append(data[:, i+bias:i+self.sample_size])

            if self.target_chars:
                rows = (self.raw_data['StimulusCode'][epoch_num][idx] == self.row_dict[self.target_chars[epoch_num]]).astype(int)
                cols = (self.raw_data['StimulusCode'][epoch_num][idx] == self.col_dict[self.target_chars[epoch_num]]).astype(int)

                target = rows + cols
            else:
                target = self.raw_data['StimulusType'][epoch_num][idx]

            X.append(res)
            y.append(target)

        self._X = np.concatenate(X)
        self._y = np.concatenate(y)

    def get_data(self):
        data = self.scaler.fit_transform(self._X)
        data, target = self.to_tensor(data, self._y)

        return data, target

    def unscale(self, X):
        return self.scaler.inverse_transform(X)