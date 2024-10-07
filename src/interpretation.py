import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphLoader

class CNN_sample:
    def __init__(self, x, y):
        self.x = x.unsqueeze(0)
        self.y = y.unsqueeze(0)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

        return self


def get_random_sample(dataset, learning_params, label=1):
    if learning_params['model_type'] == 'GNN':
        sample = next(iter(GraphLoader(dataset, batch_size=1)))
        sample.y = (torch.arange(0, learning_params['num_classes']) == label).float().unsqueeze(0)
        sample.x = torch.randn(dataset[0].x.size())
    elif learning_params['model_type'] == 'CNN':
        x = torch.randn(dataset[0][0].size())
        y = (torch.arange(0, learning_params['num_classes']) == label).float()
        sample = CNN_sample(x, y)
    else:
        raise ValueError(f"no such model type: {learning_params['model_type']}")
    
    return sample


def generate_sample(model, init_sample, criterion, learning_params, num_steps = 1001, device='cpu'):    
    model.eval()
    model.requires_grad_(False)

    init_sample = init_sample.to(device)
    init_sample.x = init_sample.x.requires_grad_()
    
    optimizer = optim.Adam([init_sample.x], lr=learning_params['lr'])
    #scheduler = StepLR(optimizer, step_size=learning_params['step_size'], gamma=learning_params['gamma'])

    num_steps = 1001

    for step in range(num_steps):
        if learning_params['model_type'] == 'GNN':
            loss = criterion(model(init_sample), init_sample.y)
        elif learning_params['model_type'] == 'CNN':
            loss = criterion(model(init_sample.x), init_sample.y)
        else:
            raise ValueError(f"no such model type: {learning_params['model_type']}")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print('-' * 10)
            print('Step {}/{}'.format(step, num_steps - 1))
            print('Loss: {:.4f}'.format(loss.item()))

    return init_sample.x.detach().cpu()


def get_best_samples(model, dataloader, learning_params, top_n=4, label=1, device='cpu'):
    model.eval()
    response_list = []
    label_list = []

    for data in dataloader:
        if learning_params['model_type'] == 'GNN':
            inputs = data.to(device)
            labels = data.y.to(device)
        elif learning_params['model_type'] == 'CNN':
            inputs = data[0].to(device)
            labels = data[1].to(device)
        else:
            raise ValueError(f"no such model type: {learning_params['model_type']}")

        outputs = model(inputs)
        response_list.append(outputs[:,label].detach().cpu())

        _, true_y = torch.max(labels.data, 1)
        label_list.append(true_y.cpu())

    response_list = torch.cat(response_list)
    label_list = torch.cat(label_list)
    response_ind = response_list.argsort(descending=True)

    return response_ind[:top_n], response_list[response_ind[:top_n]], label_list[response_ind[:top_n]]
