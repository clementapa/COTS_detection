from importlib import import_module
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

def import_class(name, instantiate = None):
    
    namesplit = name.split(".")
    module = import_module(".".join(namesplit[:-1])) 
    imported_class = getattr(module, namesplit[-1])

    if imported_class:
        if instantiate is not None:
            return imported_class(**instantiate)
        else:
            return imported_class
    raise Exception ("Class {} can be imported".format(import_class))

class WrapperDataset:
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.dataset)

def reset_weights(m, fine_tune):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''

    if fine_tune : # reset only classifier weights
        classifier_layer = [l for l in m.children()][-1]
        for l in classifier_layer:
            if hasattr(l, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {l}')
                l.reset_parameters()
    else:
        for layer in m.children():
            for l in layer:
                if hasattr(l, 'reset_parameters'):
                    print(f'Reset trainable parameters of layer = {l}')
                    l.reset_parameters()

def get_subsets(dataset, train_size=0.8):

    X = [img for img, label in dataset]
    y = [label for img, label in dataset]

    SSS = StratifiedShuffleSplit(n_splits=1, random_state=1, train_size=train_size)
    train_ids, val_ids = next(iter(SSS.split(X, y)))
    train_subset = torch.utils.data.Subset(dataset, train_ids)
    val_subset = torch.utils.data.Subset(dataset, val_ids)
    
    repartition_database(train_subset, val_subset)

    return train_subset, val_subset