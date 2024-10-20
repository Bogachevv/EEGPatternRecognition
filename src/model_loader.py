import torch
from torch import nn
import torch.nn.functional as F

from models_gnn import BaseGNN, AdjLayer
from losses import HybridLoss, GraphLoss
import regularization

from omegaconf import OmegaConf
from enum import Enum


class ModelType(Enum):
    BaseGNN = 1,

class AdjLayerInit(Enum):
    Random = 1,

class BaseCriterionType(Enum):
    MSE = 1,
    GraphLoss = 2,

class RegularizationType(Enum):
    L1 = 1,
    Nuclear = 2,
    Acyclic = 3,


def load_adj_layer(config):
    init_type = config.adj_init
    if isinstance(init_type, str):
        init_type = AdjLayerInit[init_type]

    if init_type != AdjLayerInit.Random:
        raise NotImplementedError("Only Random was implemented")

    adj_layer = AdjLayer(
        n_channels=config.n_channels,
        init_mat=None,
        is_sym=config.adj_sym
    )

    return adj_layer


def load_model(config):
    model_type = config.model
    if isinstance(model_type, str):
        model_type = ModelType[model_type]
    
    if model_type != ModelType.BaseGNN:
        raise NotImplementedError("Only BaseGNN was implemented")
    
    adj_layer = load_adj_layer(config)
    
    model = BaseGNN(
        input_feat_dim=config.input_feat_dim,
        n_channels=config.n_channels,
        adj=adj_layer,
    )

    return model


def load_criterion(config) -> HybridLoss:
    base_criterion_type = config.base_criterion
    if isinstance(base_criterion_type, str):
        base_criterion_type = BaseCriterionType[base_criterion_type]
    
    reg_types = config.get('regularization', list())
    reg_types = [RegularizationType[reg] if isinstance(reg, str) else reg for reg in reg_types]
    gamma_ls = config.get('regularization_gamma', list())

    if len(gamma_ls) != len(reg_types):
        raise ValueError(f"Arrays regularization ({len(reg_types)}) and regularization_gamma ({len(gamma_ls)}) must be same length")

    if base_criterion is BaseCriterionType.MSE:
        base_criterion = nn.MSELoss()
    elif base_criterion is BaseCriterionType.GraphLoss:
        base_criterion = GraphLoss()
    else:
        raise NotImplementedError("Only MSE and GraaphLoss was implemented")

    criterions = [{'func': base_criterion, 'kwargs_list': ['input', 'target']}]

    for reg_type, gamma in zip(reg_types, gamma_ls):
        if reg_type is RegularizationType.L1:
            reg = regularization.L1Reg(gamma)
        elif reg_type is RegularizationType.Nuclear:
            reg = regularization.NuclearReg(gamma)
        elif reg_type is RegularizationType.Acyclic:
            reg = regularization.AcyclicReg(gamma)
        else:
            raise NotImplementedError(f"Regularization type {reg_type} is not implemented yet")
        
        criterions.append({'func': reg, 'kwargs_list': ['adj']})

    criterion = HybridLoss(criterions)
    return criterion    
