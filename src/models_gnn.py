import torch
from torch import nn
import math
from torch.nn import functional as F
from torch_geometric.nn import TopKPooling, GINConv, Sequential
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class AdjLayer(nn.Module):
    def __init__(self, n_channels: int, init_mat: torch.Tensor = None, is_sym: bool = True):
        super().__init__()

        init_mat = AdjLayer._get_init(n_channels) if init_mat is None else init_mat
        self.adj_mat = nn.Parameter(init_mat)
        self.is_sym = is_sym
    
    @staticmethod
    @torch.no_grad
    def _get_init(n_channels):
        init_mat = torch.empty((n_channels, n_channels))
        nn.init.xavier_uniform_(
            tensor=init_mat,
            gain=nn.init.calculate_gain('relu')
        )

        return init_mat

    def forward(self, x):
        if self.is_sym:
            res = torch.matmul(self.adj_mat, x) + torch.matmul(self.adj_mat.T, x)
            res = res / 2
        else:
            res = torch.matmul(self.adj_mat, x)
        
        return res


class SmallGIN(nn.Module):
    def __init__(self, in_features: int, out_features: int, adj: AdjLayer, eps: float = 0.):
        super(SmallGIN, self).__init__()
        self.eps = eps
        self.adj = adj

        self.h = nn.Linear(in_features, out_features, bias=True)

    def forward(self, input):
        output = self.adj(input) # Multiplyies by adjacency matrix on the left
        output = output + (1 + self.eps) * input
        
        output = self.h(output) # Multiplyies by weight matrix on the right
        return output


class BaseGNN(nn.Module):
    def __init__(self, 
                 input_feat_dim: int, 
                 n_channels: int, 
                 adj: nn.Module, 
                 time_kernel: int = 13, 
                 num_classes: int = 2, 
                 channel_filters: int = 1
        ):

        super(BaseGNN, self).__init__()

        self.num_classes = num_classes
        self.adj = adj

        self.gc = SmallGIN(input_feat_dim, input_feat_dim, self.adj)
        self.linear_channel = nn.Conv1d(n_channels, channel_filters, kernel_size=1, bias=True)
        self.conv = nn.Conv1d(channel_filters, 1, kernel_size=time_kernel, padding='same')
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.bn2 = nn.BatchNorm1d(1)
        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(input_feat_dim, num_classes, bias=True)
        self.sig = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.gc(x)
        x = self.bn1(x)
        x = self.linear_channel(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.sig(self.linear_output(x))
        
        return x
