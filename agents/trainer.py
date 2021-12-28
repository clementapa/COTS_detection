import os, os.path as osp
from random import random

import numpy as np
import torch
import yaml
from easydict import EasyDict
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets

import wandb

wandb.login()

from tqdm import tqdm

# # Data initialization and loading
# from data import data_transforms
from datasets.ReefDataset import ReefDataset, collate_fn
import datasets.transforms as T
import utils.callbacks as callbacks
import utils.metrics as ut_metrics
import utils.utils as utils
import utils.WandbLogger as WandbLogger


class Trainer():
    def __init__(self, config_file, logger, args):

        with open(config_file, 'r') as stream:
            try:
                self.config = EasyDict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

        if args.batch_size:
            self.config.configs.batch_size = args.batch_size

        if args.csv_file:
            self.config.data.csv_file = args.csv_file

        if args.root_path:
            self.config.data.root_path = args.root_path

        if args.it:
            self.config.configs.it = args.it

        if args.notebook:
            from tqdm.notebook import tqdm

        self.fast_dev_run = args.fast_dev_run

        self.wandb_logger = WandbLogger(project=self.config.wandb.name_project,
                                        name=self.config.wandb.get("name_run"),
                                        config=self.config)

        self.logger = logger
        ##############################
        #####  PREPARATION TRAIN #####
        ##############################
        self.logger.info("Preparation training parameters")

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device : {self.device}")
        if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))

        # Seed
        torch.manual_seed(self.config.configs.get("seed", 1))

        # Model
        self.logger.info("Model : {}".format(self.config.model.name))

        model_cls = utils.import_class(self.config.model.name)
        self.model = model_cls(**self.config.model.get('params', {}))
        self.model.to(self.device)
        # self.logger.info("Model : {}".format(self.model))
        self.wandb_logger.run.watch(self.model)

        if self.config.get('criterion'):
            self.logger.info("Loss function : {}".format(
                self.config.criterion.name))
            criterion_cls = utils.import_class(self.config.criterion.name)
            self.criterion = criterion_cls(
                **self.config.criterion.get('params', {}))

        # Optimizer
        self.logger.info("Optimizer : {}".format(self.config.optimizer.name))

        optimizer_cls = utils.import_class(self.config.optimizer.name)
        self.optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.config.optimizer.params)

        # Scheduler
        if self.config.get('scheduler'):
            self.logger.info("Scheduler : {}".format(
                self.config.scheduler.name))
            scheduler_cls = utils.import_class(self.config.scheduler.name)
            self.scheduler = scheduler_cls(self.optimizer,
                                           **self.config.scheduler.params)

        # if args.checkpoint or args.relaunch:
        #     checkpoint_file = osp.join(osp.join(self.REPO_EXPERIENCE, "train"),
        #                                "checkpoint.pth")
        #     self.logger.info("Loading checkpoint {}".format(checkpoint_file))
        #     checkpoint = torch.load(checkpoint_file)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #     self.loss = checkpoint['loss']

        #     if (args.checkpoint):
        #         self.start_epoch = checkpoint['epoch']
        #     else:
        #         self.start_epoch = 1
        # else:
        if True:
            self.start_epoch = 1

        ##############################
        #####  PREPARATION DATA #####
        ##############################
        # if self.config.configs.get("k_folds"):

        #     self.k_folds = self.config.configs.k_folds
        #     self.results_k_folds = {}

        #     logger.info(f"K-fold Cross Validation with K={self.k_folds}")

        #     train_set = datasets.ImageFolder(
        #         osp.join(args["<data_repo>"], 'train_images'))

        #     val_set = datasets.ImageFolder(
        #         osp.join(args["<data_repo>"], 'val_images'))

        #     dataset = torch.utils.data.ConcatDataset([train_set, val_set])

        # elif self.config.configs.get("dataset_repartition"):

        #     dataset = datasets.ImageFolder(
        #         osp.join(args["<data_repo>"], 'images'))

        #     train_subset, val_subset = utils.get_subsets(
        #         dataset, fold=self.config.configs.dataset_repartition)

        #     train_loader = torch.utils.data.DataLoader(
        #         utils.WrapperDataset(train_subset,
        #                              transform=data_transforms['train']),
        #         batch_size=self.config.configs.batch_size,
        #         shuffle=True,
        #         num_workers=int(args["--num_workers"]))

        #     val_loader = torch.utils.data.DataLoader(
        #         utils.WrapperDataset(val_subset,
        #                              transform=data_transforms['val']),
        #         batch_size=self.config.configs.batch_size,
        #         shuffle=False,
        #         num_workers=int(args["--num_workers"]))

        #     self.logger.info("train : {}, validation : {}".format(
        #         len(train_loader.dataset), len(val_loader.dataset)))

        #     utils.repartition_database(train_loader.dataset.dataset,
        #                                val_loader.dataset.dataset)

        # else:
        # train_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(osp.join(args["<data_repo>"],'train_images'),
        #                         transform=data_transforms['train']),
        #     batch_size=self.config.configs.batch_size, shuffle=True, num_workers=int(args["--num_workers"]))

        # val_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(osp.join(args["<data_repo>"],'val_images'),
        #                         transform=data_transforms['val']),
        #     batch_size=self.config.configs.batch_size, shuffle=False, num_workers=int(args["--num_workers"]))
        if True:
            self.logger.info(f"Reading {self.config.data.root_path}")

            train_set = ReefDataset(
                self.config.data.csv_file,
                self.config.data.root_path,
                train=True,
                transforms=T.get_transform(True),
            )
            val_set = ReefDataset(self.config.data.csv_file,
                                           self.config.data.root_path,
                                           train=False,
                                           transforms=T.get_transform(False))

            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=self.config.configs.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=args.num_workers)
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=self.config.configs.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=args.num_workers)

            self.logger.info("train : {}, validation : {}".format(
                len(train_loader.dataset), len(val_loader.dataset)))

        ##############################
        #####    ENTRAINEMENT    #####
        ##############################

        # if self.config.configs.get("k_folds"):
        #     # Define the K-fold Cross Validator
        #     kfold = StratifiedKFold(
        #         n_splits=self.k_folds,
        #         shuffle=True,
        #         random_state=self.config.configs.get("seed", 1)
        #     )  # StratifiedKFold preserve the percentage of sample for each class
        #     labels = [label for img, label in dataset]
        #     # K-fold Cross Validation model evaluation
        #     for fold, (train_ids,
        #                val_ids) in enumerate(kfold.split(dataset, labels)):
        #         self.logger.info(f'FOLD {fold}')

        #         self.fold = fold
        #         self.results_k_folds[str(self.fold)] = []
        #         utils.reset_weights(
        #             self.model, self.config.model.params.get('freeze', False))

        #         train_subset = torch.utils.data.Subset(dataset, train_ids)
        #         val_subset = torch.utils.data.Subset(dataset, val_ids)

        #         # Define data loaders for training and testing data in this fold
        #         train_loader = torch.utils.data.DataLoader(
        #             utils.WrapperDataset(train_subset,
        #                                  transform=data_transforms['train']),
        #             batch_size=self.config.configs.batch_size,
        #             shuffle=True,
        #             num_workers=int(args["--num_workers"]))

        #         val_loader = torch.utils.data.DataLoader(
        #             utils.WrapperDataset(val_subset,
        #                                  transform=data_transforms['val']),
        #             batch_size=self.config.configs.batch_size,
        #             shuffle=False,
        #             num_workers=int(args["--num_workers"]))

        #         self.logger.info("train : {}, validation : {}".format(
        #             len(train_loader.dataset), len(val_loader.dataset)))

        #         utils.repartition_database(train_loader.dataset.dataset,
        #                                    val_loader.dataset.dataset)

        #         self.train_epoch(train_loader,
        #                          val_loader,
        #                          relaunch=args["--relaunch"])

        #     self.logger.info(f'Results Fold : {self.results_k_folds}')
        #     average = 0
        #     for k, results in self.results_k_folds.items():
        #         average += max(results)
        #         self.logger.info(
        #             f'Max Accuracy Fold {k} : {max(results)} Epoch : {results.index(max(results))}'
        #         )
        #     average /= self.k_folds
        #     self.logger.info(f'Average Accuracy on {k} Fold: {average}')

        # else:
        if True:
            self.train_epoch(train_loader, val_loader, relaunch=args.relaunch)

    def train(self, epoch, train_loader):

        self.model.train()
        metrics = {}

        train_iterator = tqdm(train_loader,
                              position=1,
                              desc="Training...(loss=X.X)",
                              dynamic_ncols=True,
                              total=self.config.configs.get(
                                  'it', len(train_loader)),
                              leave=False)

        for batch_idx, (data, targets) in enumerate(train_iterator):

            images = list(image.to(self.device) for image in data)
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(l for l in loss_dict.values())

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_iterator.set_description("Training... (loss=%2.5f)" %
                                           loss.data.item())

            if batch_idx > self.config.configs.get('it', 0):
                break

            loss_dict['loss_sum'] = loss
            loss_dict = {
                'train/' + k: v.detach().cpu().numpy()
                for k, v in loss_dict.items()
            }
            self.wandb_logger.run.log(loss_dict)

            if self.fast_dev_run:
                break

        self.wandb_logger.log_images((data, targets), "train", 5)

        return loss_dict

    def validation(self, val_loader, metrics_inst):

        self.model.eval()

        valid_iterator = tqdm(val_loader,
                              position=1,
                              desc="Validating...",
                              leave=False)

        metrics = {}
        
        metrics_inst["F2_score"].reset()

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(valid_iterator):

                images = list(image.to(self.device) for image in data)
                targets = [{k: v.to(self.device)
                            for k, v in t.items()} for t in targets]

                output = self.model(images, targets)

                gt_bboxes_list = [t['boxes'].cpu().numpy() for t in targets]
                pred_bboxes_list = [
                    np.concatenate((pred['scores'].unsqueeze(1).cpu().numpy(),
                                    pred['boxes'].cpu().numpy()),
                                   axis=1) for pred in output
                ]

                metrics_inst["F2_score"].update(gt_bboxes_list,
                                                pred_bboxes_list)

                if self.fast_dev_run:
                    break

        metrics["F2_score"] = metrics_inst["F2_score"].compute() # FIXME zero F2_score
        metrics = {'validation/' + k: v for k, v in metrics.items()}

        self.wandb_logger.log_images((data, targets),
                                     "validation",
                                     5,
                                     outputs=output)
        self.wandb_logger.log_videos((data, targets),
                                     "validation")  # TODO implement

        self.wandb_logger.log_metrics(metrics)

        return metrics

    def train_epoch(self, train_loader, val_loader, relaunch=False):
        self.logger.info("Launch training, start epoch : {}".format(
            self.start_epoch))

        ##### Init Metrics validation # TODO metrics instance {"train": , "validation": } en variable de classe
        metrics_instance = {
            "F2_score": ut_metrics.F2_score_competition(compute_on_step=False)
        }

        ##### Init Early stopping
        if self.config.configs.get('early_stopping'):
            early_stopping = callbacks.EarlyStopping(
                monitor=self.config.configs.early_stopping.monitor,
                mode=self.config.configs.early_stopping.mode,
                patience=self.config.configs.early_stopping.patience,
                logger=self.logger)

        ##### Init Model checkpoint
        model_checkpoint = callbacks.ModelCheckpoint(
            monitor=self.config.configs.checkpoint.monitor,
            mode=self.config.configs.checkpoint.mode,
            run=self.wandb_logger.run,
            logger=self.logger)

        epoch_iterator = tqdm(range(self.start_epoch,
                                    self.config.configs.epoch + 1),
                              total=self.config.configs.epoch,
                              initial=self.start_epoch - 1,
                              position=0,
                              desc="Epoch",
                              leave=False)

        for current_epoch in epoch_iterator:

            metrics_train = self.train(current_epoch, train_loader)

            if relaunch:
                real_epoch = current_epoch + (self.config.configs.epoch *
                                              relaunch)
            else:
                real_epoch = current_epoch

            ############################
            #####    VALIDATION    #####
            ############################
            metrics_validation = self.validation(val_loader, metrics_instance)

            metrics = metrics_train.copy()
            metrics.update(metrics_validation)

            self.wandb_logger.run.log(
                {"lr": self.optimizer.param_groups[0]['lr']})

            if self.config.get('scheduler'):
                self.scheduler.step(metrics[self.config.scheduler.get(
                    'monitor', None)])

            if hasattr(self, 'fold'):
                self.results_k_folds[str(self.fold)].append(metrics)

            ##### Update Early stopping
            if self.config.configs.get('early_stopping'):
                res = early_stopping.update(metrics)
                if res != None:
                    break

            ##### Checkpoint
            model_checkpoint.save_checkpoint(
                self,
                metrics,
                real_epoch,
                fold=self.fold if hasattr(self, 'fold') else None)

            model_checkpoint.save_weights(
                self.model, metrics, real_epoch,
                self.fold if hasattr(self, 'fold') else None)

        model_checkpoint.save_checkpoint(
            self,
            metrics,
            real_epoch,
            fold=self.fold if hasattr(self, 'fold') else None,
            name="last_checkpoint",
            end=True)