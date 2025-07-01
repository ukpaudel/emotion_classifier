import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from emotion_classifier.utils.add_noise_snr import _add_noise_with_snr
import random
from .data_augmentations import apply_augmentations 

class CREMADataset(Dataset):
    '''
    creates indexable dataset from CREMA-D https://github.com/CheyneyComputerScience/CREMA-D
    '''
    def __init__(self, dir_link, sample_rate=16000, is_train=True, noise_dir=None):
        self.dir_link = dir_link
        self.sample_rate = sample_rate
        self.noise_dir = noise_dir
        self.error_log_path = os.path.join(dir_link, "bad_files.log")
        os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)
        self.actor_ids = [] 
        self.audio_files = []
        self._collect_audio_files()

        self.crema_map = {
            'NEU': 'Neutral', 'HAP': 'Happy', 'SAD': 'Sad',
            'ANG': 'Angry', 'FEA': 'Fearful', 'DIS': 'Disgust'
        }

        self.emotion_map_ravdess = {
            1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad',
            5: 'Angry', 6: 'Fearful', 7: 'Disgust', 8: 'Surprised'
        }

        self.is_train = is_train

    def _collect_audio_files(self):
        for root, _, files in os.walk(self.dir_link):
            for file in files:
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(root, file))
                    actor_id = file.split('_')[0]
                    self.actor_ids.append(actor_id)

    def __len__(self):
        return len(self.audio_files)
    
    def _map_crema_to_ravdess(self, key_from_crema):
        label = self.crema_map.get(key_from_crema)
        return next((k for k, v in self.emotion_map_ravdess.items() if v == label), None)

    def _resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            return resampler(waveform)
        return waveform

    def _extract_label_and_audio(self, file):
        basename = os.path.basename(file)
        nameonly = os.path.splitext(basename)[0]
        labels_parts = nameonly.split('_')
        
        emotion_crema = labels_parts[2] 
        emotion = self._map_crema_to_ravdess(emotion_crema)
        
        if emotion is None:
            raise ValueError(f"Could not map emotion '{emotion_crema}' from file {file}")

        waveform, sample_rate = torchaudio.load(file)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._resample_if_needed(waveform, sample_rate)
        
        return waveform, int(emotion) - 1

    def __getitem__(self, index):
        file = self.audio_files[index]
        try:
            waveform, emotion = self._extract_label_and_audio(file)
            
            # Apply augmentations only if in training mode
            if self.is_train:
                # Call the external augmentation function
                waveform = apply_augmentations(waveform, self.sample_rate, self.noise_dir)
            
            return waveform.squeeze(0).detach(), emotion
        
        except Exception as e:
            print(f'Error with file {file}, error: {e}')
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{file} | Error: {str(e)}\n")
            return None

    def get_emotion_name(self, label):
        return self.emotion_map_ravdess.get(label + 1)