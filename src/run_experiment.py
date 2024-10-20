import model_loader
import data_preparation
import eval
import train

import torch

from omegaconf import OmegaConf
from enum import Enum

import pickle


def run(cfg_path):
    config = OmegaConf.load(cfg_path)

    model = model_loader.load_model(config)
    criterion = model_loader.load_criterion(config)
    optimizer = model_loader.load_optimizer(config, model=model)

    train_dataset, test_dataset = data_preparation.load_dataset(config)
    train_dataloader, test_dataloader = data_preparation.get_dataloaders(config, train_dataset, test_dataset)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    train_stats, val_stats = train.train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config.num_epoch,
        is_binary=config.is_binary,
        device=device,
        run_name=config.run_name,
    )

    if config.get('dump_path', None):
        with open(config['dump_path'], 'wb') as f:
            pickle.dump({
                'train': train_stats,
                'validation': val_stats,
            })