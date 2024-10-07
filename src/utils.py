import numpy as np
import time
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import Scaler

import torch
from sklearn.metrics import f1_score  
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

from losses import GraphLoss

import wandb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


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


def get_motor_subject(subject = 1):
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs, verbose=False, update_path=True)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames], verbose=False)
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                      exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=False)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    return epochs_train.get_data(), labels

def to_tensor(X_train, y_train):
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    return X_train.float(), y_train.float()

def get_cursor_data(info):
    X = []
    y = []
    for _ in tqdm(range(1, 110)):
        t_X, t_y = get_motor_subject()
        X.append(t_X)
        y.append(t_y)

    X = np.concatenate(X)
    y = np.concatenate(y)
    X, y = to_tensor(X, y)

    scaler = Scaler(info)

    X = scaler.fit_transform(X)

    return X, y


def validate_model(model, dataloader, is_binary=True, device='cpu'):
    model = model.to(device)
    model.eval()

    running_corrects = 0
    if is_binary:
        running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

    for data in dataloader:
        inputs = data[0].to(device)
        labels = data[1].to(device)
        inputs_size = inputs.size(0)
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        _, true_y = torch.max(labels.data, 1)

        if is_binary:
            P = torch.sum(preds)
            N = torch.sum(1 - preds)
            TP = torch.sum(torch.masked_select(true_y, preds == 1))
            TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
            FP = P - TP
            FN = N - TN

        running_corrects += torch.sum(preds == true_y)
        if is_binary:
            running_TP += TP
            running_TN += TN
            running_FP += FP
            running_FN += FN

    acc = running_corrects.double() / len(dataloader.dataset)

    if is_binary:
        precision = running_TP.double() / (running_TP + running_FP) if running_TP + running_FP != 0 else torch.tensor(0)
        recall = running_TP.double() / (running_TP + running_FN) if running_TP + running_FN != 0 else torch.tensor(0)
        f1 = (2 * (precision * recall) / (precision + recall)) if precision + recall != 0 else torch.tensor(0)
        bc = (recall + running_TN.double() / (running_TN + running_FP)) / 2

    min_acc, max_acc = proportion_confint(running_corrects.cpu(), len(dataloader.dataset), 0.05)
    acc = {'Accuracy': acc.cpu().data, 'Corrects': running_corrects.cpu().data, 'Min Accuracy': min_acc, 'Max Accuracy': max_acc}
    if is_binary:
        acc['Balanced Accuracy'] = bc
        acc['F1-score'] = f1
    return acc


def _run_train_epoch(model, train_dataloader, optimizer, criterion, learning_params, is_binary=True, device='cpu'):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    if is_binary:
        running_ones = 0
        running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

    for data in train_dataloader:
        if learning_params['model_type'] == 'GNN':
            inputs = data.to(device)
            labels = data.y.to(device)
            inputs_size = inputs.x.size(0)
        elif learning_params['model_type'] == 'CNN':
            inputs = data[0].to(device)
            labels = data[1].to(device)
            inputs_size = inputs.size(0)
        else:
            raise ValueError(f"no such model type: {learning_params['model_type']}")

        optimizer.zero_grad()
        if isinstance(criterion, GraphLoss):
            outputs, adj = model(inputs)
            loss = criterion(outputs, adj, labels, inputs)
        else:
            outputs = model(inputs)
            loss = criterion(input=outputs, target=labels, adj=model.adj.base_mat)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            _, true_y = torch.max(labels.data, 1)

            if is_binary:
                P = torch.sum(preds)
                N = torch.sum(1 - preds)
                TP = torch.sum(torch.masked_select(true_y, preds == 1))
                TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
                FP = P - TP
                FN = N - TN

                running_ones += P
                running_TP += TP
                running_TN += TN
                running_FP += FP
                running_FN += FN
        
            running_loss += loss.item() * inputs_size
            running_corrects += torch.sum(preds == true_y)


    epoch_loss = running_loss / len(train_dataloader.dataset) 
    epoch_acc = running_corrects.double() / len(train_dataloader.dataset)

    if is_binary:
        epoch_ones = running_ones.double() / (len(train_dataloader.dataset)  // train_dataloader.batch_size)
        epoch_precision = running_TP.double() / (running_TP + running_FP) if running_TP + running_FP != 0 else torch.tensor(0)
        epoch_recall = running_TP.double() / (running_TP + running_FN) if running_TP + running_FN != 0 else torch.tensor(0)
        epoch_f1 = (2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)) if epoch_precision + epoch_recall != 0 else torch.tensor(0)
        epoch_bc = (epoch_recall + running_TN.double() / (running_TN + running_FP)) / 2

    min_acc, max_acc = proportion_confint(running_corrects.cpu(), len(train_dataloader.dataset), 0.05)

    logs_data = {
        'train/loss': epoch_loss,
        'train/min_acc': min_acc,
        'train/epoch_acc': epoch_acc,
        'train/max_acc': max_acc,
    }
    if is_binary:
        logs_data['train/epoch_bc'] = epoch_bc
        logs_data['train/epoch_ones'] = epoch_ones
        logs_data['train/epoch_precision'] = epoch_precision
        logs_data['train/epoch_recall'] = epoch_recall
        logs_data['train/epoch_f1'] = epoch_f1

    if hasattr(model, 'adj') and hasattr(model.adj.base_mat, 'requires_grad') and model.adj.base_mat.requires_grad:
        A_grad = model.adj.base_mat.grad.detach()
        logs_data['train/A_grad'] = torch.linalg.norm(A_grad)

    wandb.log(data=logs_data)


@torch.no_grad
def _run_val_epoch(model, val_dataloader, criterion, learning_params, is_binary=True, device='cpu'):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    if is_binary:
        running_ones = 0
        running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

    for data in val_dataloader:
        if learning_params['model_type'] == 'GNN':
            inputs = data.to(device)
            labels = data.y.to(device)
            inputs_size = inputs.x.size(0)
        elif learning_params['model_type'] == 'CNN':
            inputs = data[0].to(device)
            labels = data[1].to(device)
            inputs_size = inputs.size(0)
        else:
            raise ValueError(f"no such model type: {learning_params['model_type']}")

        if isinstance(criterion, GraphLoss):
            outputs, adj = model(inputs)
            loss = criterion(outputs, adj, labels, inputs)
        else:
            outputs = model(inputs)
            loss = criterion(input=outputs, target=labels, adj=model.adj.base_mat)

        _, preds = torch.max(outputs, 1)
        _, true_y = torch.max(labels.data, 1)

        if is_binary:
            P = torch.sum(preds)
            N = torch.sum(1 - preds)
            TP = torch.sum(torch.masked_select(true_y, preds == 1))
            TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
            FP = P - TP
            FN = N - TN

            running_ones += P
            running_TP += TP
            running_TN += TN
            running_FP += FP
            running_FN += FN
        
        running_loss += loss.item() * inputs_size
        running_corrects += torch.sum(preds == true_y)

    epoch_loss = running_loss / len(val_dataloader.dataset) 
    epoch_acc = running_corrects.double() / len(val_dataloader.dataset)

    if is_binary:
        epoch_ones = running_ones.double() / (len(val_dataloader.dataset)  // val_dataloader.batch_size)
        epoch_precision = running_TP.double() / (running_TP + running_FP) if running_TP + running_FP != 0 else torch.tensor(0)
        epoch_recall = running_TP.double() / (running_TP + running_FN) if running_TP + running_FN != 0 else torch.tensor(0)
        epoch_f1 = (2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)) if epoch_precision + epoch_recall != 0 else torch.tensor(0)
        epoch_bc = (epoch_recall + running_TN.double() / (running_TN + running_FP)) / 2

    min_acc, max_acc = proportion_confint(running_corrects.cpu(), len(val_dataloader.dataset), 0.05)

    logs_data = {
        'val/loss': epoch_loss,
        'val/min_acc': min_acc,
        'val/epoch_acc': epoch_acc,
        'val/max_acc': max_acc,
    }
    if is_binary:
        logs_data['val/epoch_bc'] = epoch_bc
        logs_data['val/epoch_ones'] = epoch_ones
        logs_data['val/epoch_precision'] = epoch_precision
        logs_data['val/epoch_recall'] = epoch_recall
        logs_data['val/epoch_f1'] = epoch_f1

    wandb.log(data=logs_data)


def train_model(model, dataloaders, criterion, learning_params, is_binary=True, device='cpu', run_name: str = 'run'):
    wandb_run = wandb.init(
        project='EEGPatternRecognition',
        save_code=True,
        name=run_name,
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_params['lr'], weight_decay=learning_params['weight_decay'])
    scheduler = StepLR(optimizer, step_size=learning_params['step_size'], gamma=learning_params['gamma'])
    model = model.to(device)

    for epoch in tqdm(range(learning_params['num_epochs'])):
        _run_train_epoch(
            model=model,
            train_dataloader=dataloaders['train'],
            optimizer=optimizer,
            criterion=criterion,
            learning_params=learning_params,
            is_binary=is_binary,
            device=device
        )

        _run_val_epoch(
            model=model,
            val_dataloader=dataloaders['val'],
            criterion=criterion,
            learning_params=learning_params,
            is_binary=is_binary,
            device=device
        )

        scheduler.step()
    
    wandb_run.finish()


def plot_sample(raw_dataset, signal_sample, info, is_mean=False):
    output = raw_dataset.unscale(signal_sample.numpy())[0]

    plt.figure(figsize=(10, 10))
    mean_output = output.mean(axis=0)
    t_axis = np.arange(len(mean_output)) / info['sfreq'] * 1000
    plt.plot(t_axis, mean_output)
    plt.ylabel('amplitude (muV)')
    plt.xlabel('time (ms)')
    plt.title('Averaged EEG signal')
    plt.show()

    mne_output = mne.io.RawArray(output, info=info, verbose=False)
    plt.figure(figsize=(10, 10))
    mne_output.plot(
                n_channels=len(info['ch_names']), scalings='auto', 
                title='Raw EEG signal')
    plt.show()


def show_progress(loss, metric, loss_title, metric_title):
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(15)

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(loss)), loss, 'r', linewidth=2)

    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('mean loss', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(loss_title, fontsize=14)
    
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(metric[metric_title])), metric[metric_title], 'b', linewidth=2)

    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('criterion', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(metric_title, fontsize=14)

    plt.grid()
    plt.show()


def paired_proportions_exact_test(preds_a, preds_b, targets):
    preds_a = preds_a == targets
    preds_b = preds_b == targets

    a = sum((preds_a == 1) & (preds_b == 1))
    b = sum((preds_a == 1) & (preds_b == 0))
    c = sum((preds_a == 0) & (preds_b == 1))
    d = sum((preds_a == 0) & (preds_b == 0))
    print([[a, b], [c, d]])

    return mcnemar([[a, b], [c, d]], exact=True).pvalue


def infer_model(model, dataloader, channel=None, device='cpu', model_type='CNN'):
    model = model.to(device)
    model.eval()
    all_preds = []

    for data in dataloader:
        if model_type == 'GNN':
            inputs = data.to(device)
            labels = data.y.to(device)
            inputs_size = inputs.x.size(0)
        elif model_type == 'CNN':
            inputs = data[0].to(device)
            labels = data[1].to(device)
            inputs_size = inputs.size(0)
        else:
            raise ValueError(f"no such model type: {model_type}")

        with torch.no_grad():
            if channel is not None:
                inputs = inputs[:, channel].unsqueeze(1)
            outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)

        all_preds.append(outputs)
    return torch.cat(all_preds).cpu()
