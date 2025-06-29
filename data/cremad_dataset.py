# ravdess_dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset

class CREMADataset(Dataset):
    '''
    creates indexable dataset from CREMA-D  https://github.com/CheyneyComputerScience/CREMA-D
    '''
    def __init__(self, dir_link, sample_rate=16000):
        self.dir_link = dir_link
        self.sample_rate = sample_rate
        self.error_log_path = os.path.join(dir_link, "bad_files.log")
        os.makedirs(dir_link, exist_ok=True)

        self.audio_files = []
        self._collect_audio_files()

        self.crema_map = {
            'NEU': 'Neutral', 'HAP': 'Happy', 'SAD': 'Sad',
            'ANG': 'Angry', 'FEA': 'Fearful', 'DIS': 'Disgust'
        }
        #[ActorID]_[SentenceID]_[Emotion]_[EmotionIntensity].[extension]

        self.emotion_map_ravdess = {
            1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad',
            5: 'Angry', 6: 'Fearful', 7: 'Disgust', 8: 'Surprised'
        }


    def _collect_audio_files(self):
        for root, _, files in os.walk(self.dir_link):
            for file in files:
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.audio_files)
    
    def _map_crema_to_ravdess(self, key_from_crema):
        """
        Maps a CREMA emotion key to its corresponding numeric RAVDESS label.
        
        Args:
            key_from_crema (str): A key like 'NEU', 'HAP', etc. from crema_map.
            crema_map (dict): Dictionary mapping CREMA keys to emotion labels.
            ravdess_map (dict): Dictionary mapping numeric RAVDESS keys to emotion labels.
        
        Returns:
            int or None: Corresponding RAVDESS label number, or None if not found.
        """
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
        labels = nameonly.split('_')
        _, _, emotion_crema, _ = labels
        emotion = self._map_crema_to_ravdess(emotion_crema)
        #modality, voice, emotion, intensity, statement, repetition, actor = labels #returns string

        waveform, sample_rate = torchaudio.load(file)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

        waveform = self._resample_if_needed(waveform, sample_rate)
        return waveform, int(emotion) - 1  # Zero-indexed emotion

    def __getitem__(self, index):
        file = self.audio_files[index]
        try:
            waveform, emotion = self._extract_label_and_audio(file)
            return waveform, emotion
        except Exception as e:
            print(f'Error with file {file}, error: {e}')
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{file} | Error: {str(e)}\n")
            return None

    def get_emotion_name(self, label):
        return self.emotion_map[label + 1]  # Adjust for 0-indexing
