'''
    Author: Clément APAVOU
'''
import torch
import os, os.path as osp
import wandb

class EarlyStopping():
    def __init__(self, monitor, mode, patience, logger):
        """
        mode min ou max the metric to monitor 
        """
        self.monitor = monitor
        self.mode = mode
        self.value_keep = 1000 if self.mode == "min" else 0
        self.trigger_time = 0
        self.patience = patience
        self.logger = logger

    def update(self, metrics):

        assert self.monitor in metrics, f"{self.monitor} not a metric"
        value = metrics[self.monitor]
        if self.mode == "max":
            if value < self.value_keep:
                self.trigger_time += 1
                self.logger.info(f'trigger time early stopping : {self.trigger_time}')

                if self.trigger_time >= self.patience:
                    self.logger.info('Early stopping!')
                    return "stop"
            else:
                self.trigger_time = 0

                self.logger.info(f'trigger time early stopping : {self.trigger_time}')

                self.value_keep = value
        else:
            if value > self.value_keep:
                self.trigger_time += 1
                self.logger.info(f'trigger time early stopping : {self.trigger_time}')

                if self.trigger_time >= self.patience:
                    self.logger.info('Early stopping!')
                    return "stop"
            else:
                self.trigger_time = 0

                self.logger.info(f'trigger time : {self.trigger_time}')

                self.value_keep = value


class ModelCheckpoint():
    def __init__(self, monitor, mode, run, logger):
        self.monitor = monitor
        self.mode = mode
        self.value_keep = 1000 if self.mode == "min" else 0
        self.logger = logger
        self.run = run

    def save_checkpoint(self,
                        trainer,
                        metrics,
                        epoch,
                        fold=None,
                        name="checkpoint",
                        end=False):
        if fold is not None: name += f"_fold_{fold}"

        if end:
            self._save(trainer, epoch, "last_checkpoint")
            self.logger.info("Save last checkpoint")
            return

        value = self._check_metric_name(metrics)

        if self.mode == "max":
            if value >= self.value_keep:
                self.logger.info(
                    f"Save checkpoint {self.monitor}: new value {value} >= {self.value_keep}"
                )
                self._save(trainer, epoch, name)
                self.value_keep = value
        else:
            if value <= self.value_keep:
                self.logger.info(
                    f"Save checkpoint {self.monitor}: new value {value} <= {self.value_keep}"
                )
                self._save(trainer, epoch, name)
                self.value_keep = value

    def save_weights(self,
                     model,
                     metrics,
                     epoch=None,
                     fold=None,
                     name="weight"):
        if epoch is not None: name += f"_{epoch}"
        if fold is not None: name += f"_fold_{fold}"

        value = self._check_metric_name(metrics)

        if self.mode == "max":
            if value >= self.value_keep:
                self.logger.info(
                    f"Save weights {self.monitor}: new value {value} >= {self.value_keep}"
                )
                torch.save(model.state_dict(),
                           osp.join(self.run.dir, f"{name}.pth"))
                wandb.save(osp.join(self.run.dir, f"{name}.pth"), base_path=self.run.dir)
                self.value_keep = value
        else:
            if value <= self.value_keep:
                self.logger.info(
                    f"Save weights {self.monitor}: new value {value} <= {self.value_keep}"
                )
                torch.save(model.state_dict(),
                           osp.join(self.run.dir, f"{name}.pth"))
                wandb.save(osp.join(self.run.dir, f"{name}.pth"), base_path=self.run.dir)
                self.value_keep = value

    def _check_metric_name(self, metrics):

        assert self.monitor in metrics, f"{self.monitor} not a metric"

        return metrics[self.monitor]

    def _save(self, trainer, epoch, name):

        dict_save = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict()
        }

        if hasattr(trainer, 'criterion'): dict_save['loss'] = trainer.criterion
        if hasattr(trainer, 'scheduler'):
            dict_save['scheduler_state_dict'] = trainer.scheduler.state_dict()

        torch.save(dict_save, osp.join(self.run.dir, f"{name}.pth"))
        wandb.save(osp.join(self.run.dir, f"{name}.pth"), base_path=self.run.dir)