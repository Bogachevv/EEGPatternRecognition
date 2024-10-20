import torch
from torch import nn
from torch.nn import functional as F


class GraphLoss(nn.Module):
    def __init__(self, base, alpha=1, beta=1, gamma=1):
        super(GraphLoss, self).__init__()
        self.base = base
        self.alpha = alpha 
        self.beta = beta
        self.gamma = gamma

    def forward(self, predictios, adj_matrix, labels, inputs):
        predictios = predictios.view(-1)
        labels = labels.view(-1)
        n = adj_matrix.size(2)

        base_loss = self.base(predictios, labels) # F.mse_loss
        
        #degree_matrix = adj_matrix.sum(dim=1)
        #laplacian = degree_matrix - adj_matrix
        #laplacian
        smoothness_term = torch.mean(torch.cdist(inputs, inputs, p=2).pow(2) * adj_matrix, dim=(1, 2)) / 2

        connectivity_term = - torch.log(adj_matrix.sum(dim=1)).mean(dim=1)

        sparsity_term = torch.norm(adj_matrix, dim=(1, 2)) / (n*n)

        graph_loss = self.alpha * smoothness_term + self.beta * connectivity_term + self.gamma * sparsity_term
        
        return base_loss + graph_loss.mean()


class HybridLoss(nn.Module):
    def __init__(self, loss_desc_ls: list[dict]) -> None:
        super().__init__()

        self.loss_ls = [
            HybridLoss._get_loss_callable(desc)
            for desc in loss_desc_ls
        ]


    @staticmethod
    def _get_loss_callable(loss_desc):
        def func(*args, **kwargs):
            if 'kwargs_list' in loss_desc:
                kw_filtered = {
                    key: kwargs[key]
                    for key in loss_desc['kwargs_list']
                }
            else:
                kw_filtered = kwargs
            
            return loss_desc['func'](*args, **kw_filtered)

        return func
        
    
    def forward(self, *args, **kwargs):
        s = 0
        for loss in self.loss_ls:
            s = s + loss(*args, **kwargs)
        
        return s
