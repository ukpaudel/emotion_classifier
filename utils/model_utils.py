import torch
import os
import yaml
from torchaudio import save


def save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path, logger=None, scheduler=None):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc
    }
    if scheduler:
        checkpoint['scheduler_state'] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    if logger:
        logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, logger=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']

    if logger:
        logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {start_epoch}")

    return start_epoch, best_val_acc

def save_misclassified_audio(waveforms, labels, predicted, lengths, config, epoch, logger=None, file_paths=None):
    #the emotion_map vs file number are shifted by 1. Emotion map 0 = neutral = file index 01
    EMOTION_MAP = {
        0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad',
        4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'
    }
    log_dir = config['logging']['log_dir']
    run_name = config['logging'].get('run_label', log_dir)
    class_folder= os.path.join("misclassified",log_dir,run_name)
    os.makedirs(class_folder, exist_ok=True)
    sample_rate = config["data"].get("sample_rate", 16000)

    for i in range(len(labels)):
        true_label = EMOTION_MAP.get(labels[i].item(), str(labels[i].item()))
        pred_label = EMOTION_MAP.get(predicted[i].item(), str(predicted[i].item()))
        file_info = os.path.basename(file_paths[i]) if file_paths else f"sample{i}"

        filename = f"{class_folder}/epoch{epoch}_{file_info}_true{true_label}_pred{pred_label}.wav"
        if true_label != pred_label:
            try:
                waveform = waveforms[i].cpu()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                save(filename, waveform, sample_rate)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save misclassified audio: {e}")

