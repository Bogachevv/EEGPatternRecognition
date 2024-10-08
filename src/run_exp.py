import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import scipy

from sym_linear import SymLinear
from models_gnn import BaseGNN
from utils import train_model
from losses import HybridLoss

def run(
    run_name_fmt: str,
    reg_cls: nn.Module,
    gamma_grid: np.ndarray,
    A_init: torch.Tensor,
    data_loaders,
    learning_params,
    device
):
    criterion = nn.MSELoss()

    res_mat = []

    for gamma in gamma_grid:
        torch.manual_seed(44)
        np.random.seed(44)

        A = SymLinear(A_init.clone())
        A.requires_grad_(True)
        model_GNN = BaseGNN(48, 64, A)
        hybrid_criterion = HybridLoss([
            {'func': criterion, 'kwargs_list': ['input', 'target']},
            {'func': reg_cls(gamma=gamma), 'kwargs_list': ['adj']}
        ])

        train_model(
            model=model_GNN, 
            dataloaders=data_loaders, 
            criterion=hybrid_criterion, 
            learning_params=learning_params, 
            device=device, 
            run_name=run_name_fmt.format(f"{gamma:.5f}")
            )
        
        res_mat.append(
            model_GNN.adj.base_mat.detach().cpu()
        )

    return res_mat