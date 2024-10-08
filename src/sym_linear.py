import torch
from torch import nn
import torch.nn.functional as F

class SymLinear(nn.Module):
    def __init__(self, mat: torch.Tensor):
        super().__init__()

        self.base_mat = nn.Parameter(mat)
    
    def forward(self, x):
        res = torch.matmul(self.base_mat, x) + torch.matmul(self.base_mat.T, x)
        res = res / 2

        return res