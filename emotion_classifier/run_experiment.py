import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynndescent")

import os
import sys
import torch
from utils.config import load_config
from utils.logger import setup_logger
from utils.run_tracker import update_model_runs_yaml
from utils.tensorboard_plot_utils import plot_from_tensorboard
from utils.confusion_animation import animate_confusion, plot_final_confusion
from data.dataloader import create_dataloaders
from models.emotion_model import EmotionModel
from train.trainer import train_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_experiment(config_path="configs/config.yml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.cfg['device'] = device

    log_dir = config['logging']['log_dir']
    run_name = config['logging'].get('run_label', log_dir)
    logger = setup_logger(log_dir, run_name)
    logger.info("===== Running Experiment =====")

    if config['logging'].get('track_run', True):
        run_config_path = config['logging'].get('runs_config_path', 'configs/model_runs.yml')
        update_model_runs_yaml(run_config_path, log_dir, run_name)

    model = EmotionModel(
        encoder_name=config.get("model", "encoder_name"),
        dropout=config.get("model", "dropout"),
        hidden_dim=config.get("model", "hidden_dim"),
        num_classes=config.get("model", "num_classes"),
        freeze_encoder=config.get("model", "freeze_encoder"),
        unfreeze_last_n_layers=config.get("model", "unfreeze_last_n_layers", default=None),
        logger=logger
    )

    from utils.latent_visualizer import register_hooks   # centralize hooks there
    register_hooks(model)

    train_loader, val_loader = create_dataloaders(config)

    resume_training = config['training'].get('resume_training', False)
    train_model(model, train_loader, val_loader, config, run_name, resume_training=resume_training)

    plot_from_tensorboard(config_path)

    animate_confusion(os.path.join(log_dir, run_name))
    plot_final_confusion(os.path.join(log_dir, run_name))

    # optional
    logger.info("===== Latent visualization can now be run separately via visualize_latents.py =====")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment from config")
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to YAML config file")
    args = parser.parse_args()

    run_experiment(args.config)
