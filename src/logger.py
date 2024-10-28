from enum import Enum
from abc import ABC, abstractmethod

import wandb
import wandb.wandb_run


class LoggerType(Enum):
    nologger = 0,
    console = 1,
    wandb = 2


class Logger(ABC):
    def __init__(self, project: str, run_name: str):
        self.project = project
        self.run_name = run_name

    @abstractmethod
    def log(self, data: dict):
        pass

    @abstractmethod
    def finish(self):
        pass


class WandbLogger(Logger):
    def __init__(self, project, run_name, **kwargs):
        super().__init__(project, run_name)

        self._run: wandb.wandb_run.Run = wandb.init(
            project=project,
            name=run_name,
            **kwargs
        )

    def log(self, data: dict):
        self._run.log(data)
    
    def finish(self):
        self._run.finish()


class ConsoleLogger(Logger):
    def __init__(self, project, run_name, **kwargs):
        super().__init__(project, run_name)

    def log(self, data: dict):
        print(*(
            f'{key}: {val}'
            for key, val in data.items()
        ), sep='\t', end='\n\n')
    
    def finish(self):
        pass


class NoLogger(Logger):
    def __init__(self, project, run_name, **kwargs):
        super().__init__(project, run_name)
        pass

    def log(self, data: dict):
        pass
    
    def finish(self):
        pass
