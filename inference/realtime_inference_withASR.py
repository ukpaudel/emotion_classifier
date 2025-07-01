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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

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

class RealTimeEmotionAndASR:
    def __init__(self, emotion_model, sample_rate, device):
        self.device = device
        self.emotion_model = emotion_model.to(self.device)
        self.sample_rate = sample_rate
        self.window_size = 5 * sample_rate
        self.step_size = int(sample_rate * 0.05)
        self.buffer = np.zeros(self.window_size, dtype=np.float32)

        self.total_samples_seen = 0
        self.samples_since_last_classify = 0
        self.classify_interval = int(sample_rate * 5)
        self.averaging_chuncks = int(self.window_size/self.classify_interval)

        # load ASR model

        self.asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        print(self.asr_processor)

        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(self.device)
        self.asr_model.eval()
        self.current_text = ""

        print(f"[INFO] Using device: {self.device}")

        self.emotion_map = {
            0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad',
            4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'
        }
        self.current_probs = np.zeros(len(self.emotion_map))
        self.current_emotion = "Neutral"
        self.label_history = deque(maxlen=30)
        self.need_classify = False

        # plot
        self.fig, (self.ax_bar, self.ax_wave) = plt.subplots(2, 1, figsize=(8, 4))

        # emotion bar
        self.bar = self.ax_bar.bar(self.emotion_map.values(), self.current_probs)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_ylabel("Probability")
        self.ax_bar.set_title("Real-time Emotion Probabilities")
        self.ax_bar.tick_params(axis='x', rotation=45)

        # waveform
        self.wave_line, = self.ax_wave.plot([], [], lw=1)
        self.ax_wave.set_title("Rolling Waveform with Emotion + ASR")
        self.ax_wave.set_xlabel("Downsampled Time")

        # pinned emotion text
        self.emotion_text_handle = self.ax_wave.text(
            0.95, 0.9, "", transform=self.ax_wave.transAxes,
            fontsize=14, color="red", ha="right", va="center"
        )

        # pinned ASR text
        self.asr_text_handle = self.ax_wave.text(
            0.95, 0.8, "", transform=self.ax_wave.transAxes,
            fontsize=12, color="blue", ha="right", va="center"
        )

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        samples = indata[:, 0]
        self.buffer = np.roll(self.buffer, -len(samples))
        self.buffer[-len(samples):] = samples

        self.samples_since_last_classify += len(samples)
        self.total_samples_seen += len(samples)

        if self.samples_since_last_classify >= self.classify_interval:
            self.need_classify = True
            self.samples_since_last_classify = 0

    def classify_and_transcribe(self):
        buffer_rms = np.sqrt(np.mean(self.buffer ** 2))
        if buffer_rms < 0.01:
            print(f"[INFO] Skipping classification and ASR (low RMS={buffer_rms:.5f})")
            self.current_probs = 0*self.current_probs
            self.current_text = ""
            self.current_emotion = "Silence"

            return

        # ASR
        inputs = self.asr_processor(self.buffer, return_tensors="pt", sampling_rate=self.sample_rate)
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self.asr_model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.asr_processor.decode(pred_ids[0])
        self.current_text = transcription
        print(f"[ASR] {transcription}")

        # EMOTION
        chunk_size = self.sample_rate
        chunk_preds = []
        for i in range(self.averaging_chuncks ):
            start = i * chunk_size
            end = start + chunk_size
            chunk = self.buffer[start:end]
            waveform = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            lengths = torch.tensor([waveform.shape[1]]).to(self.device)
            with torch.no_grad():
                output = self.emotion_model(waveform, lengths)
                probs = F.softmax(output, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                chunk_preds.append(pred_idx)
        majority_class = Counter(chunk_preds).most_common(1)[0][0]
        self.current_emotion = self.emotion_map[majority_class]
        self.current_probs = probs.squeeze().cpu().numpy()
        print(f"[EMOTION] Aggregated emotion over 5s: {self.current_emotion}")

        self.label_history.append((self.current_emotion, self.total_samples_seen))

    def update_plot(self, frame):
        if self.need_classify:
            self.classify_and_transcribe()
            self.need_classify = False

        # emotion bar
        for bar, p in zip(self.bar, self.current_probs):
            bar.set_height(p)

        # waveform
        downsample_factor = 100
        ds_wave = self.buffer[::downsample_factor]
        x_vals = np.arange(len(ds_wave))
        self.wave_line.set_ydata(ds_wave)
        self.wave_line.set_xdata(x_vals)
        self.ax_wave.relim()
        self.ax_wave.autoscale_view()

        # update emotion pinned text
        self.emotion_text_handle.set_text(self.current_emotion)

        # update ASR pinned text
        self.asr_text_handle.set_text(self.current_text)

        return self.bar, self.wave_line, self.emotion_text_handle, self.asr_text_handle

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
    emotion_model, target_sr = load_model_from_config("configs/inference_config.yml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    realtime = RealTimeEmotionAndASR(emotion_model, target_sr, device=device)
    realtime.run()
