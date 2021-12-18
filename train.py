"""
Script to launch the training 

Usage:
    train.py <yaml_file> [options]

Arguments:
    <yaml_file>     Configuration train file 
    <data_repo>     folder where data is located. train_images/ and val_images/ need to be found in the folder

Options:
    -h                                  Display help
    --log = LEVEL_LOG                   Level of log [default: DEBUG]
    --experiment = REPO                 folder where experiment outputs are located [default: Experiment]
    --log-interval = IT                 how many batches to wait before logging training status 
    --checkpoint                        If the training crashed and you want to relaunch with checkpoint
    --relaunch = NB_FOIS_RELAUNCH       If you want to relaunch the training to do more epoch
    --num_workers = W                   [default: 1]
    --visualise_data
    --remove
"""

import yaml
import os, os.path as osp
from docopt import docopt
from easydict import EasyDict
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision
from sklearn.model_selection import StratifiedKFold

import wandb
wandb.login()

from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.logger import init_logger

# # Data initialization and loading
# from data import data_transforms
import datasets.dataset as Datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable warning cuda for tensorboard

class Trainer():

    def __init__(self, config_file):

        with open(config_file, 'r') as stream:
            try:
                self.config = EasyDict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

        self.REPO_EXPERIENCE = osp.join(args["--experiment"], config_file.split('/')[-1].split('.')[0])  
        if(not args["--checkpoint"] and not args["--relaunch"]):
            remove = False
            # Create experiment folder
            if not os.path.isdir(self.REPO_EXPERIENCE):
                os.makedirs(self.REPO_EXPERIENCE)
            else :
                if ars['--remove']:
                    remove = True
                else:
                    logger.warning("The repository " + self.REPO_EXPERIENCE + " already exists !")
                    if (input("The repository " + self.REPO_EXPERIENCE + " will be delete. If you don't want to continue CTRL + C")!='n'):
                        remove = True
                if remove :
                    shutil.rmtree(self.REPO_EXPERIENCE)
                    logger.warning("The repository " + self.REPO_EXPERIENCE + " was deleted")
                else:
                    exit()

            os.makedirs(osp.join(self.REPO_EXPERIENCE, "train"), exist_ok = True)
            os.makedirs(osp.join(self.REPO_EXPERIENCE, "weight"), exist_ok = True)
        
            shutil.copy(config_file, osp.join(self.REPO_EXPERIENCE, "train"))

        self.run = wandb.init(project=self.config.wandb.name_project,
                                name=config_file.split('/')[-1].split('.')[0])

        ##############################
        #####  PREPARATION TRAIN #####
        ##############################
        logger.info("Preparation training parameters")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device : {self.device}")
        if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
        
        # Seed
        torch.manual_seed(self.config.configs.get("seed", 1))

        # Model
        logger.info("Model : {}".format(self.config.model.name))

        model_cls = utils.import_class(self.config.model.name)
        self.model = model_cls(**self.config.model.get('params', {}))
        self.model.to(self.device)
        logger.info("Model : {}".format(self.model))

        if self.config.get("task") == "object_detection":
            pass
        else:
            # Loss function
            logger.info("Loss function : {}".format(self.config.criterion.name))
            criterion_cls = utils.import_class(self.config.criterion.name)
            self.criterion = criterion_cls(**self.config.criterion.get('params', {}))

        # Optimizer
        logger.info("Optimizer : {}".format(self.config.optimizer.name))

        optimizer_cls = utils.import_class(self.config.optimizer.name)
        self.optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config.optimizer.params)    

        # Scheduler
        if self.config.get('scheduler') :
            logger.info("Scheduler : {}".format(self.config.scheduler.name))
            scheduler_cls = utils.import_class(self.config.scheduler.name)
            self.scheduler = scheduler_cls(self.optimizer, **self.config.scheduler.params)

        if(args["--checkpoint"] or args["--relaunch"]):
            checkpoint_file = osp.join(osp.join(self.REPO_EXPERIENCE, "train"), "checkpoint.pth")
            logger.info("Loading checkpoint {}".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.loss = checkpoint['loss']

            if(args["--checkpoint"]):
                self.start_epoch = checkpoint['epoch']
            else:
                self.start_epoch = 1
        else:
            self.start_epoch = 1

        ##############################
        #####  PREPARATION DATA #####
        ##############################
        if self.config.configs.get("k_folds"):

            self.k_folds = self.config.configs.k_folds
            self.results_k_folds = {}

            logger.info(f"K-fold Cross Validation with K={self.k_folds}")

            train_set = datasets.ImageFolder(osp.join(args["<data_repo>"],'train_images'))

            val_set = datasets.ImageFolder(osp.join(args["<data_repo>"],'val_images'))

            dataset = torch.utils.data.ConcatDataset([train_set, val_set])
        
        elif self.config.configs.get("dataset_repartition"):
            
            dataset = datasets.ImageFolder(osp.join(args["<data_repo>"], 'images'))

            train_subset, val_subset = utils.get_subsets(dataset,
                                                        fold = self.config.configs.dataset_repartition)

            train_loader = torch.utils.data.DataLoader(
                            utils.WrapperDataset(train_subset, transform=data_transforms['train']), 
                            batch_size=self.config.configs.batch_size, 
                            shuffle=True,
                            num_workers=int(args["--num_workers"]))

            val_loader = torch.utils.data.DataLoader(
                            utils.WrapperDataset(val_subset, transform=data_transforms['val']),
                            batch_size=self.config.configs.batch_size, 
                            shuffle=False,
                            num_workers=int(args["--num_workers"]))

            logger.info("train : {}, validation : {}".
                    format(len(train_loader.dataset), len(val_loader.dataset)))
            
            utils.repartition_database(train_loader.dataset.dataset, val_loader.dataset.dataset)
        
        else:
            # train_loader = torch.utils.data.DataLoader(
            #     datasets.ImageFolder(osp.join(args["<data_repo>"],'train_images'),
            #                         transform=data_transforms['train']),
            #     batch_size=self.config.configs.batch_size, shuffle=True, num_workers=int(args["--num_workers"]))

            # val_loader = torch.utils.data.DataLoader(
            #     datasets.ImageFolder(osp.join(args["<data_repo>"],'val_images'),
            #                         transform=data_transforms['val']),
            #     batch_size=self.config.configs.batch_size, shuffle=False, num_workers=int(args["--num_workers"]))
            logger.info(f"Reading {self.config.data.root_path}")

            train_set = Datasets.ReefDataset(self.config.data.csv_file, self.config.data.root_path, train=True, transform=Datasets.get_transform(True))
            val_set = Datasets.ReefDataset(self.config.data.csv_file, self.config.data.root_path, train=False, transform=Datasets.get_transform(False))
            
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.configs.batch_size,
                                                        shuffle=False, collate_fn=Datasets.collate_fn,
                                                        num_workers=int(args["--num_workers"]))
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config.configs.batch_size, 
                                                        shuffle=False, collate_fn=Datasets.collate_fn,
                                                        num_workers=int(args["--num_workers"]))

            logger.info("train : {}, validation : {}".
                    format(len(train_loader.dataset),len(val_loader.dataset)))

        ##############################
        #####    ENTRAINEMENT    #####
        ##############################
        
        if self.config.configs.get("k_folds"):
            # Define the K-fold Cross Validator
            kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.config.configs.get("seed", 1)) # StratifiedKFold preserve the percentage of sample for each class
            labels = [label for img, label in dataset]
            # K-fold Cross Validation model evaluation
            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset, labels)):
                logger.info(f'FOLD {fold}')

                self.fold = fold
                self.results_k_folds[str(self.fold)] = []
                utils.reset_weights(self.model, self.config.model.params.get('freeze', False))

                train_subset = torch.utils.data.Subset(dataset, train_ids)
                val_subset = torch.utils.data.Subset(dataset, val_ids)

                # Define data loaders for training and testing data in this fold
                train_loader = torch.utils.data.DataLoader(
                                utils.WrapperDataset(train_subset, transform=data_transforms['train']), 
                                batch_size=self.config.configs.batch_size, 
                                shuffle=True,
                                num_workers=int(args["--num_workers"]))

                val_loader = torch.utils.data.DataLoader(
                                utils.WrapperDataset(val_subset, transform=data_transforms['val']),
                                batch_size=self.config.configs.batch_size, 
                                shuffle=False,
                                num_workers=int(args["--num_workers"]))


                logger.info("train : {}, validation : {}".
                        format(len(train_loader.dataset), len(val_loader.dataset)))

                utils.repartition_database(train_loader.dataset.dataset, val_loader.dataset.dataset)

                self.train_epoch(train_loader, val_loader, relaunch = args["--relaunch"])

            logger.info(f'Results Fold : {self.results_k_folds}')
            average = 0
            for k, results in self.results_k_folds.items(): 
                average += max(results)
                logger.info(f'Max Accuracy Fold {k} : {max(results)} Epoch : {results.index(max(results))}')
            average /= self.k_folds
            logger.info(f'Average Accuracy on {k} Fold: {average}')

        else:
            # self.writer = SummaryWriter(log_dir = osp.join(self.REPO_EXPERIENCE, "train"))
            self.train_epoch(train_loader, val_loader, relaunch = args["--relaunch"])
            # self.writer.close()

    def train(self, epoch, train_loader):
        
        self.model.train()
        train_loss = 0
        correct = 0

        for batch_idx, (data, targets) in enumerate(train_loader): 
            
            self.optimizer.zero_grad()

            if self.config.get("task") == "object_detection":
                images = list(image.to(self.device) for image in data)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] # probleme si image avec 0 bounding boxes

                loss_dict = self.model(images, targets)
                loss = sum(l for l in loss_dict.values())
            else:
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, targets)
            
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

            # backpropagation
            loss.backward()
            self.optimizer.step()

            # metrics
            train_loss += loss.data.item()

            if args["--log-interval"]:
                if batch_idx % int(args["--log-interval"]) == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()))

        train_loss /= len(train_loader)
        logger.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

        return train_loss, correct.item()/len(train_loader.dataset) 

    def validation(self, val_loader):

        self.model.eval()
        validation_loss = 0
        correct = 0

        with torch.no_grad():
            for data, targets in val_loader: #tqdm(val_loader, position = 1, desc = "Validation", leave = False):
            
                if self.config.get("task") == "object_detection":
                    images = list(image.to(self.device) for image in data)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] # probleme si image avec 0 bounding boxes

                    loss_dict = self.model(images, targets)
                    loss = sum(l for l in loss_dict.values())
                else:
                    data, targets = data.to(self.device), targets.to(self.device)
                    output = self.model(data)

                    validation_loss += self.criterion(output, targets).data.item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader)
        logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

        return validation_loss, correct.item()/len(val_loader.dataset)

    def train_epoch(self, train_loader, val_loader, relaunch = False):
        logger.info("Launch training, start epoch : {}".format(self.start_epoch))

        losses_train = []
        accuracies_train = []
        acc_val_previous = 0

        if self.config.configs.get('early_stopping'):
            trigger_times = 0
            loss_val_previous = 100

        for current_epoch in tqdm(range(self.start_epoch,self.config.configs.epoch+1), total = self.config.configs.epoch , initial = self.start_epoch -1 , position = 0, desc = "Epoch", leave = False):
            
            loss_train, acc_train = self.train(current_epoch, train_loader)
            
            if self.config.get('scheduler') :
                self.scheduler.step()

            losses_train.append(loss_train) 
            accuracies_train.append(acc_train) 

            if relaunch :
                real_epoch = current_epoch + (self.config.configs.epoch * int(relaunch))
            else:
                real_epoch = current_epoch
            
            if not self.config.get('checkpoint'):
                        ############################
                        #####    VALIDATION    #####
                        ############################               
                loss_val, acc_val = self.validation(val_loader)
                
                if acc_val >= acc_val_previous:
                    logger.info(f"Save weights at the epoch {real_epoch}")

                    if self.config.configs.get('k_folds'): 
                        self.save_weights(fold = self.fold)
                    else:
                        self.save_weights()

                    if self.config.configs.get('k_folds'): 
                        self.results_k_folds[str(self.fold)].append(acc_val)
                    else:
                        self.update_wandb(real_epoch, losses_train[-1], accuracies_train[-1], loss_val = loss_val, accuracy_val = acc_val)
                    acc_val_previous = acc_val
                else:
                    if not self.config.configs.get('k_folds'): self.update_wandb(real_epoch, losses_train[-1], accuracies_train[-1], loss_val = loss_val, accuracy_val = acc_val)
                
                ##### Early stopping
                if self.config.configs.get('early_stopping'):
                    if loss_val > loss_val_previous : 
                        trigger_times +=1
                        logger.info(f'trigger times : {trigger_times}')

                        if trigger_times >= self.config.configs.early_stopping.params.patience:
                            logger.info('Early stopping!')
                            return trigger_times
                    else:
                        logger.info('trigger times : 0')
                        trigger_times = 0

                    loss_val_previous = loss_val
            else:
                if current_epoch % self.config.checkpoint.get('save_weight_step', 0.1) == 0:
                    if self.config.configs.get('k_folds'):
                        self.save_weights(fold = self.fold)
                    else:
                        self.save_weights()

                if current_epoch  % self.config.checkpoint.get('checkpoint_step', 0.1) == 0:
                    if self.config.configs.get('k_folds'):
                        self.checkpoint(real_epoch, fold = self.fold)
                    else:
                        self.checkpoint(real_epoch)
        
        if self.config.configs.get('k_folds'):
            self.checkpoint(real_epoch, fold = self.fold)
        else:
            self.checkpoint(real_epoch)

    def update_tensorboard(self, current_epoch, loss_train, accuracy_train, loss_val= None, accuracy_val = None):
        pass
        # if not self.config.get('checkpoint'):  
        #     # self.writer.add_scalar('Loss/train', loss_train, current_epoch-1)
        #     # self.writer.add_scalar('Loss/test', loss_val, current_epoch-1)
        #     # self.writer.add_scalar('Accuracy/train', accuracy_train, current_epoch-1)
        #     # self.writer.add_scalar('Accuracy/test', accuracy_val, current_epoch-1)
        # elif current_epoch % self.config.checkpoint.get('test_step', 0.1) == 0:  
        #     # self.writer.add_scalar('Loss/train', loss_train, current_epoch-1)
        #     # self.writer.add_scalar('Loss/test', loss_val, current_epoch-1)
        #     # self.writer.add_scalar('Accuracy/train', accuracy_train, current_epoch-1)
        #     # self.writer.add_scalar('Accuracy/test', accuracy_val, current_epoch-1)
        # else:
        #     # self.writer.add_scalar('Loss/train', loss_train, current_epoch-1)
        #     # self.writer.add_scalar('Accuracy/train', accuracy_train, current_epoch-1)
    
    def update_wandb(self, current_epoch, loss_train, accuracy_train, loss_val= None, accuracy_val = None):
        self.run.log()
        if not self.config.get('checkpoint'):  
            self.run.log({'train_loss': loss_train, 'epoch': current_epoch-1})
            self.run.log({'val_loss': loss_val, 'epoch': current_epoch-1})
            self.run.log({'train_accuracy': accuracy_train, 'epoch': current_epoch-1})
            self.run.log({'val_accuracy': accuracy_val, 'epoch': current_epoch-1})
        elif current_epoch % self.config.checkpoint.get('test_step', 0.1) == 0:  
            self.run.log({'train_loss': loss_train, 'epoch': current_epoch-1})
            self.run.log({'val_loss': loss_val, 'epoch': current_epoch-1})
            self.run.log({'train_accuracy': accuracy_train, 'epoch': current_epoch-1})
            self.run.log({'val_accuracy': accuracy_val, 'epoch': current_epoch-1})
        else:
            self.run.log({'train_loss': loss_train, 'epoch': current_epoch-1})
            self.run.log({'train_accuracy': accuracy_train, 'epoch': current_epoch-1})

    def save_weights(self, epoch = None, fold = None):
        name = "weight"
        if epoch is not None: name += f"_{epoch}"
        if fold is not None: name += f"_fold_{fold}"

        torch.save(self.model.state_dict(), osp.join(osp.join(self.REPO_EXPERIENCE, "weight"), f"{name}.pth"))

    def checkpoint(self, epoch, fold = None):
        name = "checkpoint"
        if fold is not None: name += f"_fold_{fold}"

        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.criterion},
                    osp.join(osp.join(self.REPO_EXPERIENCE, "train"), f"{name}.pth"))
                    # 'scheduler_state_dict': self.scheduler.state_dict(),

if __name__ == '__main__':
    args = docopt(__doc__) 

    logger = init_logger("Train", args['--log'])

    Trainer(args['<yaml_file>'])