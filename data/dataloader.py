import os
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from data.ravdess_dataset import RAVDESSDataset
from data.collate import collate_fn

import importlib

def resolve_class(class_path):
    '''utility to dynamically load the class'''
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_dataloaders(config):
    '''
    This function will merge different datasets and outputs train and test dataset. 
    It takes the input about dataset class and dataset location from the input. Example config file
    #config.yml
    data:
    datasets:
        - class_path: data.ravdess_dataset.RAVDESSDataset
        data_dir: 'data/ravdess/data'
        - class_path: data.cremad_dataset.CREMADataset
        data_dir: 'data/CREMA-D/data'
    '''
    datasets_config = config['data']['datasets']
    val_split = config['training'].get('val_split', 0.2)
    batch_size = config['training'].get('batch_size', 8)
    num_workers = config['training'].get('num_workers', 2)

    all_datasets = []

    for ds_cfg in datasets_config:
        DatasetClass = resolve_class(ds_cfg['class_path'])
        dataset = DatasetClass(ds_cfg['data_dir'])

        # Optional: clean nulls
        if hasattr(dataset, 'audio_files'):
            dataset.audio_files = [f for f in dataset.audio_files if f is not None]

        all_datasets.append(dataset)

    combined_dataset = ConcatDataset(all_datasets)

    val_size = int(len(combined_dataset) * val_split)
    train_size = len(combined_dataset) - val_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader
