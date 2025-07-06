import torch
import torchaudio
import importlib
import yaml
import torch.nn.functional as F
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque, Counter
import os
from utils.emotion_labels import EMOTION_MAP


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def resolve_class(class_path_or_callable):
    if callable(class_path_or_callable):
        return class_path_or_callable
    module_path, class_name = class_path_or_callable.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_model_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    Classifier = resolve_class(config['model']['class_path'])
    model = Classifier(**config['model']['args'])
    checkpoint = torch.load(config['model']['checkpoint_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, config['model']['sample_rate']

class RealTimeEmotion:
    def __init__(self, model, sample_rate, device):
        self.device = device
        self.model = model.to(self.device)
        self.sample_rate = sample_rate
        self.window_size = 5 * sample_rate      # 5s window
        self.step_size = int(sample_rate * 0.05) # 50ms blocksize
        self.buffer = np.zeros(self.window_size, dtype=np.float32)

        self.total_samples_seen = 0   # absolute sample counter

        print(f"[INFO] Using device: {self.device}")

        self.current_probs = np.zeros(len(EMOTION_MAP))
        self.current_emotion = "Neutral"

        # store (label, absolute_sample_position)
        self.label_history = deque(maxlen=30)

        self.classify_interval = int(sample_rate * 1)  # classify every 1s
        self.samples_since_last_classify = 0

        # plot setup
        self.fig, (self.ax_bar, self.ax_wave) = plt.subplots(2, 1, figsize=(12, 6))

        # emotion bar
        self.bar = self.ax_bar.bar(EMOTION_MAP.values(), self.current_probs)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_ylabel("Probability")
        self.ax_bar.set_title("Real-time Emotion Probabilities")
        self.ax_bar.tick_params(axis='x', rotation=45)

        # waveform
        self.wave_line, = self.ax_wave.plot([], [], lw=1)
        self.ax_wave.set_title("Rolling Waveform with Emotion Timeline")
        self.ax_wave.set_xlabel("Downsampled Time")

        self.text_labels = []

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        samples = indata[:, 0]
        self.buffer = np.roll(self.buffer, -len(samples))
        self.buffer[-len(samples):] = samples

        self.samples_since_last_classify += len(samples)
        self.total_samples_seen += len(samples)

        if self.samples_since_last_classify >= self.classify_interval:
            self.predict_current_window()
            self.samples_since_last_classify = 0

    def predict_current_window(self):
        buffer_rms = np.sqrt(np.mean(self.buffer ** 2))
        if buffer_rms < 0.01:
            print(f"[INFO] Skipping classification (low RMS={buffer_rms:.5f})")
            self.label_history.append((None, self.total_samples_seen))
            return

        # break 5s buffer into 5×1s chunks
        chunk_size = self.sample_rate
        chunk_preds = []

        for i in range(5):
            start = i * chunk_size
            end = start + chunk_size
            chunk = self.buffer[start:end]

            waveform = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            lengths = torch.tensor([waveform.shape[1]]).to(self.device)

            with torch.no_grad():
                output = self.model(waveform, lengths)
                probs = F.softmax(output, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                chunk_preds.append(pred_idx)

        # majority vote
        majority_class = Counter(chunk_preds).most_common(1)[0][0]
        majority_label = EMOTION_MAP[majority_class]

        # also update current_probs from the final chunk to refresh barplot
        self.current_probs = probs.squeeze().cpu().numpy()
        self.current_emotion = majority_label
        print(f"Aggregated emotion over 5s: {majority_label}")

        self.label_history.append((majority_label, self.total_samples_seen))

    def update_plot(self, frame):
        for bar, p in zip(self.bar, self.current_probs):
            bar.set_height(p)

        downsample_factor = 100
        ds_wave = self.buffer[::downsample_factor]
        x_vals = np.arange(len(ds_wave))
        self.wave_line.set_ydata(ds_wave)
        self.wave_line.set_xdata(x_vals)

        self.ax_wave.relim()
        self.ax_wave.autoscale_view()

        for txt in self.text_labels:
            txt.remove()
        self.text_labels.clear()

        for label, abs_pos in self.label_history:
            if label is None:
                continue
            rel_pos_samples = abs_pos - (self.total_samples_seen - self.window_size)
            rel_pos_downsampled = rel_pos_samples // downsample_factor
            if 0 <= rel_pos_downsampled < len(ds_wave):
                txt = self.ax_wave.text(
                    rel_pos_downsampled,
                    0.8 * self.ax_wave.get_ylim()[1],
                    label,
                    fontsize=10,
                    color="red",
                    ha="center",
                    va="center"
                )
                self.text_labels.append(txt)

        return self.bar, self.wave_line, self.text_labels

    def run(self):
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.step_size,
            device=3  # change if needed
        ):
            ani = FuncAnimation(self.fig, self.update_plot, interval=50)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    model, target_sr = load_model_from_config("configs/inference_config.yml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    realtime = RealTimeEmotion(model, target_sr, device=device)
    realtime.run()
