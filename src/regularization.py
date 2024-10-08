import torch
from torch import nn
import torch.nn.functional as F


class L1Reg(nn.Module):
    def __init__(self, gamma) -> None:
        super().__init__()
        self.gamma = gamma
    
    def forward(self, adj):
        adj_l1 = torch.linalg.norm(adj.reshape((-1, )), ord=1)

        return self.gamma * adj_l1


class NuclearReg(nn.Module):
    def __init__(self, gamma) -> None:
        super().__init__()
        self.gamma = gamma
    
    def forward(self, adj):
        adj_l1 = torch.linalg.norm(adj, ord='nuc')

        return self.gamma * adj_l1

  
class AcyclicReg(nn.Module):
    def __init__(self, gamma) -> None:
        super().__init__()
        self.gamma = gamma
    
    def forward(self, adj):
        adj_sq = torch.square(adj)
        adj_exp = torch.linalg.matrix_exp(adj_sq)
        reg_val = torch.trace(adj_exp) - adj.shape[0]

        return self.gamma * reg_val