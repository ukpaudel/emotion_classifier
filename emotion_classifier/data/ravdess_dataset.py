import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from utils.add_noise_snr import _add_noise_with_snr
import random
from tqdm import tqdm
from .data_augmentations import apply_augmentations 

class RAVDESSDataset(Dataset):
    '''
    creates indexable dataset from RAVDESS

    The dataset implements on-the-fly data augmentation during training. 
    This means that for each audio file retrieved during a training epoch, 
    random transformations are applied to the waveform.
    '''
    def __init__(self, dir_link, sample_rate=16000, is_train=True, noise_dir=None):
        self.dir_link = dir_link
        self.sample_rate = sample_rate
        self.noise_dir = noise_dir
        self.dataset_name = 'ravdess'
        self.error_log_path = os.path.join(dir_link, "bad_files.log")
        os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)
        self.actor_ids = [] 
        self.audio_files = []
        self._collect_audio_files()

        self.emotion_map = {
            1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad',
            5: 'Angry', 6: 'Fearful', 7: 'Disgust', 8: 'Surprised'
        }

        self.is_train = is_train

        self.noise_file_paths_map = noise_dir
        if self.is_train and self.noise_dir and os.path.isdir(self.noise_dir):
            self._preload_noise_file_paths() # New method to preload noise paths
  
    def _preload_noise_file_paths(self):
        """Pre-collects all noise file paths for faster access during __getitem__."""
        self.noise_file_paths_map = {}
        noise_categories = ['noise', 'speech', 'music'] # Define these centrally
        
        for category in noise_categories:
            category_path = os.path.join(self.noise_dir, category)
            if os.path.isdir(category_path):
                # IMPORTANT: Include all relevant audio extensions
                self.noise_file_paths_map[category] = [
                    os.path.join(category_path, f)
                    for f in os.listdir(category_path)
                    if f.lower().endswith(('.wav', '.flac', '.mp3')) # Adjust as per your noise files
                ]
            else:
                tqdm.write(f"Warning: Noise category directory not found: {category_path}. Skipping this category.")
                self.noise_file_paths_map[category] = [] # Ensure it's an empty list

    def _collect_audio_files(self):
        for root, _, files in os.walk(self.dir_link):
            for file in files:
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(root, file))
                    actor_id = file.split('-')[-1].split('.')[0]
                    self.actor_ids.append(actor_id)


    def __len__(self):
        return len(self.audio_files)

    def _resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            return resampler(waveform)
        return waveform

    def _extract_label_and_audio(self, file):
        basename = os.path.basename(file)
        nameonly = os.path.splitext(basename)[0]
        labels_parts = nameonly.split('-')
        emotion_raw = labels_parts[2] 

        try:
            emotion_label = int(emotion_raw)
        except ValueError:
            raise ValueError(f"Could not parse emotion ID from filename: {file}. Expected an integer at labels_parts[2]. Got: '{emotion_raw}'")

        if emotion_label not in self.emotion_map:
             raise ValueError(f"Emotion label {emotion_label} from file {file} not found in emotion_map.")

        waveform, sample_rate = torchaudio.load(file)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._resample_if_needed(waveform, sample_rate)
        
        return waveform, emotion_label - 1 

    def __getitem__(self, index):
        file = self.audio_files[index]
        try:
            waveform, emotion = self._extract_label_and_audio(file)
            # Determine the device (e.g., 'cuda' if available, else 'cpu')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # ---  Move waveform to GPU BEFORE augmentations ---
            if device.type == 'cuda': # Only move if a GPU is actually available
                waveform = waveform.to(device) 

            # Apply augmentations only if in training mode
            if self.is_train:
                # Call the external augmentation function
                #TODO Kwargs for the data augmentation
                #waveform = waveform
                waveform = apply_augmentations(waveform, self.sample_rate,  self.noise_file_paths_map, device)
            # Return the processed waveform (now on GPU) and label
            # The .squeeze(0) converts [1, N] to [N] if needed
            return waveform.squeeze(0).detach(), emotion, self.dataset_name
        
        except Exception as e:
            tqdm.write(f'Error with file {file}, error: {e}') 
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{file} | Error: {str(e)}\n")
            return None 

    def get_emotion_name(self, label):
        return self.emotion_map.get(label + 1)