# ravdess_dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset

class RAVDESSDataset(Dataset):
    '''
    creates indexable dataset from RAVDESDataset
    '''
    def __init__(self, dir_link, sample_rate=16000):
        self.dir_link = dir_link
        self.sample_rate = sample_rate
        self.error_log_path = os.path.join(dir_link, "bad_files.log")
        os.makedirs(dir_link, exist_ok=True)

        self.audio_files = []
        self._collect_audio_files()

        self.emotion_map = {
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

    def _resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            return resampler(waveform)
        return waveform

    def _extract_label_and_audio(self, file):
        basename = os.path.basename(file)
        nameonly = os.path.splitext(basename)[0]
        labels = nameonly.split('-')
        _, _, emotion, _, _, _, _ = labels
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
