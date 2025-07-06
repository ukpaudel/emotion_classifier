import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns
import io
from tqdm import tqdm
from utils.logger import setup_logger
from utils.model_utils import save_checkpoint, load_checkpoint, save_misclassified_audio
from utils.run_tracker import update_model_runs_yaml
from sklearn.metrics import confusion_matrix
from utils.emotion_labels import EMOTION_MAP

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #this is a patch I hade to make to plot as I have dll conflicts. 
"""
trainer.py

Handles training and validation loop with support for:
- Resume training from checkpoint
- TensorBoard logging
- Misclassified audio saving
- Flexible optimizer/loss/scheduler

Assumes: model implements forward(x, lengths)
"""

def train_model(model, train_loader, val_loader, config, run_name, resume_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device} will be used for the training')
    model = model.to(device)

    # === SETUP PATHS ===
    log_dir = os.path.join(config['logging']['log_dir'], run_name)
    checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger(log_dir=log_dir, run_name=run_name)
    writer = SummaryWriter(log_dir=log_dir)

    # === SETUP OPTIMIZER AND SCHEDULER ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    if config['training']['scheduler'] == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['max_lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['training']['epochs'],
            pct_start=0.1
        )
    elif config['training']['scheduler'] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get("min_lr", 1e-6)
        )
    elif config['training']['scheduler'] == "cosine_warm_restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=config['training'].get("min_lr", 1e-6)
        )
    else:
        scheduler = None  # or raise error

    
    if logger:
        datasets_config = config['data']['datasets']
        for ds_cfg in datasets_config:
            classname = ds_cfg['class_path']
            path = ds_cfg['data_dir']
            logger.info(f"[INFO] Data used for trainings are  {classname} from {path}")
    
    # === CASE 1: RESUME FULL TRAINING ===
    start_epoch = 0
    best_val_acc = 0.0

    if resume_training:
        if os.path.exists(checkpoint_path):
            start_epoch, best_val_acc = load_checkpoint(
                model, optimizer, scheduler, checkpoint_path, device, logger
            )
            if logger:
                logger.info(f"[INFO] Resumed training from epoch {start_epoch}")
        else:
            if logger:
                logger.warning(f"[WARN] resume_training=True but checkpoint not found at {checkpoint_path}")
                logger.info("[INFO] Starting fresh training")

    # === CASE 2: FRESH TRAINING BUT LOAD PRETRAINED ENCODER ===
    elif config.get("pretrained_weights").get("enabled", False):
        pretrained_path = config["pretrained_weights"].get("checkpoint_path")
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            encoder_state_dict = {
                k.replace("encoder.", ""): v
                for k, v in checkpoint["model_state"].items()
                if k.startswith("encoder.")
            }
            model.load_state_dict(checkpoint["model_state"], strict=False)
            #TODO if we have unfreezed a layer this doesn't work.
            if config['model'].get('freeze_encoder'):
                for param in model.encoder.parameters():
                    param.requires_grad = False

            model.encoder.load_state_dict(encoder_state_dict, strict=False)
            logger.info(f"[INFO] Loaded encoder weights from {pretrained_path}")
        else:
            logger.warning(f"[WARN] Pretrained checkpoint not found: {pretrained_path}")
            logger.info("[INFO] Proceeding without encoder preloading.")

    # === CASE 3: FRESH TRAINING AND CLEAN LOGS ===
    else:
        if os.path.exists(log_dir):
            logger.info(f"[INFO] Removing old log directory: {log_dir}")
            shutil.rmtree(log_dir, ignore_errors=True)
            os.makedirs(log_dir, exist_ok=True)

        logger.info("[INFO] Starting fresh training")

    # === LET THE TRAINING BEGIN ===
    all_confusions = {}
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Using tqdm for a nice progress bar
        # Ensure that 'total' is correct if you have multiple datasets concatenated
        # The total length of the loader is based on the number of batches
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{config['training']['epochs']} Training")

        for i, batch_data in pbar:
            # --- ADD THIS CHECK HERE ---
            if batch_data is None: 
                # This means the collate_fn received a batch where all samples were None
                # (i.e., all files in that batch failed to load or process).
                tqdm.write(f"Skipping empty batch at epoch {epoch+1}, batch {i+1} as all samples failed.")
                msg = f"[WARN] Skipping empty batch at epoch {epoch+1}, batch {i+1} as all samples failed."
                if logger:
                    logger.warning(msg)
                continue # Skip to the next iteration of the loop
            # --- END OF CHECK ---

            waveforms, labels, lengths, dataset_ids = batch_data 
            # Move data to the correct device (GPU if available)
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(waveforms, lengths)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            #backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=total_loss / (i + 1))
            
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # === VALIDATION ===
        model.eval()
        val_correct, val_total = 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for waveforms, labels, lengths, dataset_ids in val_loader:
                waveforms, labels, lengths, dataset_ids = waveforms.to(device), labels.to(device), lengths.to(device), dataset_ids.to(device)
                outputs = model(waveforms, lengths)
                _, predicted = torch.max(outputs, 1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # accumulate for confusion
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Optional: save misclassified examples at the end of the epoch
                if config['logging'].get("save_misclassified") and epoch==config['training']['epochs']-1:
                    save_misclassified_audio(waveforms, labels, predicted, lengths, config, epoch, logger)

        # calculate confusion matrix and convert it to a tensorboard image and log it
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8,8))
        emotion_labels = [EMOTION_MAP[i] for i in range(len(EMOTION_MAP))]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=emotion_labels,
                    yticklabels=emotion_labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix Epoch {epoch+1}")

        # convert to tensorboard image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = torchvision.transforms.ToTensor()(image)
        writer.add_image("ConfusionMatrix", image, epoch)
        plt.close(fig)

        # save confusion matrix for this epoch into dict
        all_confusions[epoch] = cm

        # save confusion matrix as numpy file
        np.save(os.path.join(log_dir, "confusions_all_epochs.npy"), all_confusions)

        
        val_acc = 100 * val_correct / val_total
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        logger.info(f"Validation Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path, logger, scheduler)
            logger.info("New best model saved.")

    writer.close()
    logger.info("Training complete.")
