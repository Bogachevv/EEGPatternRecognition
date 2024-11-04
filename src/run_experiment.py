import model_loader
import data_preparation
import eval
import train
import logger
import utils

import torch

from omegaconf import OmegaConf
from enum import Enum

import pickle


def run(cfg_path):
    config = OmegaConf.load(cfg_path)

    utils.set_global_seed(config.seed)

    model = model_loader.load_model(config)
    criterion = model_loader.load_criterion(config)
    optimizer = model_loader.load_optimizer(config, model=model)

    adj_init = model.adj.adj_mat.detach().cpu().numpy()

    train_dataset, test_dataset = data_preparation.load_dataset(config)
    train_dataloader, test_dataloader = data_preparation.get_dataloaders(config, train_dataset, test_dataset)

    logger_type = config.get('logger', 'nologger')
    if isinstance(logger_type, str):
        logger_type = logger.LoggerType[logger_type]
    
    if logger_type is logger.LoggerType.nologger:
        lgr = logger.NoLogger(project='EEGPatternRecognition', run_name=config.run_name)
    elif logger_type is logger.LoggerType.console:
        lgr = logger.ConsoleLogger(project='EEGPatternRecognition', run_name=config.run_name)
    elif logger_type is logger.LoggerType.wandb:
        lgr = logger.WandbLogger(project='EEGPatternRecognition', run_name=config.run_name, save_code=True)
    else:
        raise ValueError('Incorrect logger type')

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
        logger=lgr,
        max_grad_norm=config.get('max_grad_norm', 1000.0),
    )

    adj_final = model.adj.adj_mat.detach().cpu().numpy()

    print(f"{train_stats=}\n\n{val_stats=}")

    if config.get('dump_path', None):
        logs = {
            'train_curve': {
                'train': dict(train_stats),
                'validation': dict(val_stats),
            },
            'adj': {
                'init': adj_init,
                'final': adj_final,
            },
            'config': OmegaConf.to_container(config, resolve=True)
        }

        with open(config['dump_path'], 'wb') as f:
            pickle.dump(obj=logs, file=f)