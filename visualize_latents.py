import torch
import os
from utils.logger import setup_logger
from data.dataloader import create_dataloaders
from models.emotion_model import EmotionModel
from utils.latent_visualizer import register_hooks, extract_features_for_visualization, plot_latent_space
from utils.config import load_config
from utils.analyze_confusion_latent import analyze_confusion_and_latent

'''
This will take weights and will run val_loader from the dataset and will visualize latent space from it. 
'''
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


def visualize_from_checkpoint(config_path="configs/config.yml", checkpoint_path=None):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.cfg['device'] = device

    log_dir = config['logging']['log_dir']
    run_name = config['logging'].get('run_label', log_dir)
    logger = setup_logger(log_dir, run_name)

    model = EmotionModel(
        encoder_name=config.get("model", "encoder_name"),
        dropout=config.get("model", "dropout"),
        hidden_dim=config.get("model", "hidden_dim"),
        num_classes=config.get("model", "num_classes"),
        freeze_encoder=config.get("model", "freeze_encoder"),
        unfreeze_last_n_layers=config.get("model", "unfreeze_last_n_layers", default=None),
        logger=logger
    )
    
    model.to(device)
    register_hooks(model)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(log_dir, run_name, "checkpoint.pt")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

   
    _, val_loader = create_dataloaders(config)
    # now rebuild val_loader with single worker so there is no fork conflict for loaded weights 
    val_loader = torch.utils.data.DataLoader(
    val_loader.dataset,
    batch_size=val_loader.batch_size,
    shuffle=False,
    collate_fn=val_loader.collate_fn,
    num_workers=0
    )
    
    extract_features_for_visualization(model, val_loader, device, logger)
    plot_latent_space(os.path.join(log_dir, run_name), logger)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    visualize_from_checkpoint(args.config, args.checkpoint)
