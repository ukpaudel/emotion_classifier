import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import GroupShuffleSplit
from data.ravdess_dataset import RAVDESSDataset
from data.cremad_dataset import CREMADataset
from data.collate import collate_fn

import importlib

def resolve_class(class_path):
    '''utility to dynamically load the class'''
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_dataloaders(config):
    '''
    This function will merge different datasets and outputs train and test dataloaders.
    It takes the input about dataset class and dataset location from the input.
    
    Example config file:
    # config.yml
    data:
      datasets:
        - class_path: data.ravdess_dataset.RAVDESSDataset
          data_dir: 'data/ravdess/data'
        - class_path: data.cremad_dataset.CREMADataset
          data_dir: 'data/CREMA-D/data'
      noise_dir: 'data/audio_noise_samples' # Optional: Path to a directory of noise files

    training:
      val_split: 0.2
      batch_size: 8
      num_workers: 2
    
    This version performs GroupShuffleSplit separately for each dataset,
    ensuring speaker independence WITHIN each dataset, then merges the splits.
    '''
    datasets_config = config['data']['datasets']
    val_split = config['training'].get('val_split', 0.2)
    batch_size = config['training'].get('batch_size', 8)
    num_workers = config['training'].get('num_workers', 2)

    # Get the global noise directory from config
    global_noise_dir = config['data'].get('noise_dir', None) 
    print('NOISE DATA from ', global_noise_dir)

    all_train_subsets = []
    all_val_subsets = []

    for ds_cfg in datasets_config:
        DatasetClass = resolve_class(ds_cfg['class_path'])
        data_dir = ds_cfg['data_dir']

        # Instantiate the full dataset temporarily to get its audio files and actor IDs
        # We pass is_train=False and noise_dir=None here, as this instance is only for metadata/splitting.
        full_dataset_for_split = DatasetClass(dir_link=data_dir, is_train=False, noise_dir=None)

        num_samples = len(full_dataset_for_split)
        if num_samples == 0:
            print(f"Warning: No valid audio files found for dataset: {ds_cfg['class_path']} at {data_dir}. Skipping.")
            continue

        # Get actor IDs for splitting. Crucially, this must be a list aligned with each sample.
        if hasattr(full_dataset_for_split, "actor_ids") and len(full_dataset_for_split.actor_ids) == num_samples:
            current_dataset_actor_ids = full_dataset_for_split.actor_ids
        else:
            print(f"Warning: Dataset {ds_cfg['class_path']} does not have a correctly aligned `actor_ids` list. Falling back to per-sample grouping for split (less ideal for speaker independence).")
            current_dataset_actor_ids = [str(i) for i in range(num_samples)]

        # Perform GroupShuffleSplit for THIS SPECIFIC dataset
        gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        local_train_idx, local_val_idx = next(gss.split(range(num_samples), groups=current_dataset_actor_ids))

        # --- Create Train Subset for this dataset ---
        # Instantiate the DatasetClass specifically for the training split,
        # passing is_train=True and the noise_dir.
        train_ds_instance = DatasetClass(dir_link=data_dir, is_train=True, noise_dir=global_noise_dir)
        all_train_subsets.append(Subset(train_ds_instance, local_train_idx))

        # --- Create Validation Subset for this dataset ---
        # Instantiate the DatasetClass specifically for the validation split,
        # passing is_train=False and no noise_dir.
        val_ds_instance = DatasetClass(dir_link=data_dir, is_train=False, noise_dir=None)
        all_val_subsets.append(Subset(val_ds_instance, local_val_idx))

    # Check if any data was collected across all datasets
    if not all_train_subsets and not all_val_subsets:
        raise ValueError("No valid audio files found across all configured datasets. Cannot create dataloaders.")

    # Concatenate all individual train subsets and validation subsets
    train_dataset = ConcatDataset(all_train_subsets)
    val_dataset = ConcatDataset(all_val_subsets)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Always shuffle training data
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return train_loader, val_loader