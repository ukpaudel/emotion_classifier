import os
import torch
from torch.utils.data import DataLoader, random_split
from data.ravdess_dataset import RAVDESSDataset
from data.collate import collate_fn

def create_dataloaders(config):
    """
    Creates training and validation DataLoaders based on the config.
    """
    data_dir = config['data']['data_dir']
    val_split = config['training'].get('val_split', 0.2)
    batch_size = config['training'].get('batch_size', 8)
    num_workers = config['training'].get('num_workers', 2)
    
    # Load dataset
    dataset = RAVDESSDataset(data_dir)
    
    # Filter out None examples
    dataset.audio_files = [f for f in dataset.audio_files if f is not None]

    # Train/validation split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader
