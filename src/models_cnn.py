import torch
import math
from torch import nn
from torch.nn import functional as F


def scaled_tanh(z):
    return 1.7159 * torch.tanh((2.0 / 3.0) * z)


def get_autoregressive_mask(size):
    """
    Returns attention mask of given size for autoregressive model.
    """
    dtype = getattr(torch, 'bool', None) or torch.uint8
    res = torch.zeros(size, size, dtype=dtype)
    for i in range(size - 1):
        res[i, i + 1:] = 1
    return res


class FlexCNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_layers=2, num_classes=2):
        super(FlexCNN, self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        input_channels = n_channels
        output_channels = input_channels * 2
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(True)
        input_channels = output_channels
        output_feat_dim = input_feat_dim

        layers = []

        for _ in range(num_layers):
            output_channels = input_channels * 2
            layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool1d(2, stride=2))
            input_channels = output_channels
            output_feat_dim = output_feat_dim // 2

        self.body = nn.Sequential(*layers)
        self.hook = nn.Identity()

        self.classifier_input_size = output_channels * output_feat_dim
        self.classifier = nn.Linear(self.classifier_input_size, num_classes, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.conv(x))

        x = self.body(x)

        x = self.hook(torch.flatten(x, 1))
        x = self.sig(self.classifier(x))
        
        return x


class EEGNet(nn.Module):
    def __init__(self, sample_len=72, in_channels=64, num_classes=2):
        super(EEGNet, self).__init__()
        self.sample_len = sample_len
        self.in_channels = in_channels

        F1 = 8
        F2 = 16
        D = 2
        
        self.conv1 = nn.Conv2d(1, F1, (1, sample_len // 2), padding = 'same')
        self.bn1 = nn.BatchNorm2d(F1, False)

        self.conv2 = nn.Conv2d(F1, D * F1, (in_channels, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(D * F1, False)
        self.pool1 = nn.AvgPool2d(1, 4)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        
        self.conv3 = nn.Conv2d(D * F1, F2, (1, D), groups=D * F1, padding = 'same')
        self.bn3 = nn.BatchNorm2d(D * F1, False)
        self.pool2 = nn.AvgPool2d(1, 8)
        self.hook = nn.Dropout(p=0.5, inplace=False)
        
        self.classifier_input_size = 72 // 24 * F2
        self.classifier = nn.Linear(self.classifier_input_size, num_classes, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.elu(self.bn2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = F.elu(self.bn3(x))
        x = torch.flatten(self.pool2(x), 1)
        x = self.hook(x)
        
        x = self.sig(self.classifier(x))
        return x


class CecottiCNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_layers=2, num_classes=2, num_fiters=10):
        super(CecottiCNN, self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(n_channels, num_fiters, kernel_size=1)
        self.conv2 = nn.Conv1d(num_fiters, 5 * num_fiters, kernel_size=13, padding = 'same')

        self.classifier_input_size = input_feat_dim * 5 * num_fiters
        self.hook = nn.Linear(self.classifier_input_size, 100, bias=True)
        self.classifier = nn.Linear(100, num_classes, bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = scaled_tanh(self.conv1(x))
        x = scaled_tanh(self.conv2(x))

        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.sig(self.classifier(x))
        
        return x


class BaseCNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, time_kernel=13, num_classes=2, channel_filters=1):
        super(BaseCNN, self).__init__()

        self.num_classes = num_classes

        self.linear_channel = nn.Conv1d(n_channels, channel_filters, kernel_size=1, bias=True)
        self.conv = nn.Conv1d(channel_filters, 1, kernel_size=time_kernel, padding='same')
        self.bn1 = nn.BatchNorm1d(1)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)

        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(input_feat_dim, num_classes, bias=True)
        self.sig = nn.Softmax()

    def forward(self, x):
        x = self.linear_channel(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.sig(self.linear_output(x))
        
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, dim, max_len=100, dropout = 0.1):
        super().__init__()
        
        self.dim = dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        #self.pe = torch.zeros(1, max_len, dim)
        #arg = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) /\
        #     torch.pow(scale, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        self.pe = torch.zeros(1, dim, max_len)

        self.pe[0, 0::2, :] = torch.sin(position * div_term).transpose(0, 1)
        self.pe[0, 1::2, :] = torch.cos(position * div_term).transpose(0, 1)
               
    def forward(self, x):
        x = x + self.pe[:,:,:x.size(2)].to(x.device)
        
        return self.dropout(x)


class BaseCNNAttn(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_classes=2, num_filters=1, dropout=0.5):
        super(BaseCNNAttn, self).__init__()

        self.num_classes = num_classes
        
        self.mask = get_autoregressive_mask(input_feat_dim)

        self.linear_channel = nn.Conv1d(n_channels, num_filters, kernel_size=1, bias=True)

        self.pos_enc = PositionalEncoder(num_filters, input_feat_dim, dropout=dropout)

        self.queries = nn.Linear(input_feat_dim, input_feat_dim)
        self.keys = nn.Linear(input_feat_dim, input_feat_dim)
        self.values = nn.Linear(input_feat_dim, input_feat_dim)

        self.attn = nn.MultiheadAttention(num_filters, 1, dropout=dropout, batch_first=True)
        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(input_feat_dim * num_filters, num_classes, bias=True)
        self.sig = nn.Softmax()

    def forward(self, x):
        x = self.linear_channel(x)
        x = self.pos_enc(x)

        queries = self.queries(x).transpose(1, 2)
        keys = self.keys(x).transpose(1, 2)
        values = self.values(x).transpose(1, 2)

        x, self.attn_weights = self.attn(queries, keys, values, attn_mask=self.mask.to(x.device))

        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.sig(self.linear_output(x))
        
        return x
    