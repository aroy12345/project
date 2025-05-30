#!/usr/bin/env python3

from typing import Optional

import wandb


class WandbLogger():
    """
    WandB logger for training Gait Controller
    """

    def __init__(
        self,
        project: str = "project",
        task: str = "task",
        path: str = "./log.dat",
        update_interval: int = 1000,
        config={},
        model=None,
        group: Optional[str] = None,
        id: Optional[str] = None
    ) -> None:
        if id is None:
            id = wandb.util.generate_id()
        self.id = id
        self.writer = wandb.init(id=self.id,
                                 resume="allow",
                                 project=project,
                                 job_type=task,
                                 config=config,
                                 group=group,
                                 # save_code=path,
                                 dir=path,
                                 sync_tensorboard=False
                                 )

        self.update_interval = update_interval
        self.writer.watch(model, log_freq=1000)
        self.last_log_train_epoch = 0

    def log_train_data(self, collect_result: dict, epoch: int) -> None:
        self.writer.log(collect_result, step=epoch)
        self.last_log_train_epoch = epoch
