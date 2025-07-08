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

'''
This is a demo code that takes a real time audio and does ASR and emotion classification.
>cd emotion_classifier 
> python inference/realtime_inference_withASR.py 
if you run into issue, make sure your package is indexed as a packaged. perform
>pip install -e . from the root emotion_classifer folder


'''
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
        self.device_audio = 12 #SPECIFY THIS. TODO LOAD FROM .yml
        self.device = device
        self.emotion_model = emotion_model.to(self.device)
        self.sample_rate = sample_rate # This is your emotion model's sample rate
        
        self.emotion_window_sec = 3.0 
        self.emotion_window_samples = int(self.emotion_window_sec * sample_rate)
        
        self.asr_window_sec = 7.0
        self.asr_window_samples = int(self.asr_window_sec * sample_rate)

        self.buffer = np.zeros(self.asr_window_samples, dtype=np.float32)
        
        self.step_size = int(sample_rate * 0.05)
        self.classify_interval_sec = 3.0
        self.classify_interval_samples = int(self.classify_interval_sec * sample_rate)

        self.total_samples_seen = 0
        self.samples_since_last_classify = 0
        
        # Load ASR model using torchaudio.pipelines
        self.asr_bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
        self.asr_model = self.asr_bundle.get_model().to(self.device)
        self.asr_model.eval()

        # Get ASR labels from the bundle (ESSENTIAL)
        self.asr_labels = self.asr_bundle.get_labels()
        # No need to instantiate CTCDecoder here, we'll do greedy decoding manually
        
        self.asr_sample_rate = self.asr_bundle.sample_rate # Get ASR model's sample rate from bundle

        self.current_text = ""

        print(f"[INFO] Using device: {self.device}")
        print(f"[INFO] ASR Model Sample Rate: {self.asr_sample_rate} Hz")


        self.current_probs = np.zeros(len(EMOTION_MAP))
        self.current_emotion = "Neutral"
        self.label_history = deque(maxlen=30)
        self.need_classify = False

        # --- Plotting Configuration ---
        self.downsample_factor = 50 
        self.use_envelope_plot = False 
        self.envelope_smoothing_window = int(sample_rate * 0.01) 

        # Plot setup
        self.fig, (self.ax_bar, self.ax_wave) = plt.subplots(2, 1, figsize=(10, 6))

        # Emotion bar plot
        self.bar = self.ax_bar.bar(EMOTION_MAP.values(), self.current_probs, color='skyblue')
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_ylabel("Probability")
        self.ax_bar.set_title("Real-time Emotion Probabilities")
        self.ax_bar.tick_params(axis='x', rotation=45)

        # Waveform plot
        self.wave_line, = self.ax_wave.plot([], [], lw=1, color='darkblue')
        self.ax_wave.set_title("Rolling Waveform with Emotion & ASR")
        self.ax_wave.set_xlabel("Time (seconds)")
        self.ax_wave.set_xlim(0, self.asr_window_sec) 
        self.ax_wave.set_ylim(-0.5, 0.5) 

        # Pinned emotion text
        self.emotion_text_handle = self.ax_wave.text(
            0.98, 0.9, "", transform=self.ax_wave.transAxes,
            fontsize=14, color="red", ha="right", va="center"
        )

        # Pinned ASR text
        self.asr_text_handle = self.ax_wave.text(
            0.02, 0.9, "", transform=self.ax_wave.transAxes,
            fontsize=12, color="blue", ha="left", va="center"
        )

        self.rms_threshold = 0.015
        self.silence_emotion_label = "Silence/No Speech"


    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        samples = indata[:, 0]
        
        self.buffer = np.roll(self.buffer, -len(samples))
        self.buffer[-len(samples):] = samples

        self.samples_since_last_classify += len(samples)
        self.total_samples_seen += len(samples)

        if self.samples_since_last_classify >= self.classify_interval_samples:
            self.need_classify = True
            self.samples_since_last_classify = 0

    def classify_and_transcribe(self):
        current_buffer_segment = self.buffer.copy()
        
        buffer_rms = np.sqrt(np.mean(current_buffer_segment ** 2))

        if buffer_rms < self.rms_threshold:
            print(f"[INFO] Skipping classification and ASR (low RMS={buffer_rms:.5f} < {self.rms_threshold:.5f})")
            self.current_probs = np.zeros(len(EMOTION_MAP))
            self.current_text = ""
            self.current_emotion = self.silence_emotion_label
            return

        # ASR Processing
        waveform = torch.from_numpy(current_buffer_segment).unsqueeze(0).to(self.device)

        # Resample if needed to ASR model's sample rate
        if self.sample_rate != self.asr_sample_rate:
            waveform = torchaudio.functional.resample(waveform, self.sample_rate, self.asr_sample_rate)

        with torch.inference_mode():
            emissions, _ = self.asr_model(waveform)
        
        # --- Manual Greedy CTC Decoding ---
        # Get the predicted token IDs by taking the argmax of emissions
        predicted_ids = torch.argmax(emissions[0], dim=-1).cpu().numpy() # emissions[0] for the single batch item

        transcription_tokens = []
        blank_idx = 0 # Assuming blank token is at index 0 for Wav2Vec2 bundle (common)

        for i, token_id in enumerate(predicted_ids):
            if token_id == blank_idx:
                continue # Skip blank tokens
            
            # Remove consecutive duplicates
            if i > 0 and token_id == predicted_ids[i-1]:
                continue
            
            transcription_tokens.append(self.asr_labels[token_id])
        
        transcription = "".join(transcription_tokens).replace("|", " ").strip()
        # --- End Manual Greedy CTC Decoding ---

        self.current_text = transcription.strip()
        print(f"[ASR] {self.current_text}")

        # EMOTION Classification
        emotion_segment = current_buffer_segment[-self.emotion_window_samples:]
        waveform = torch.from_numpy(emotion_segment).unsqueeze(0).to(self.device)
        lengths = torch.tensor([self.emotion_window_samples]).to(self.device) 
        
        with torch.no_grad():
            logits_emotion, logits_domain, pooled_encoder_features = self.emotion_model(waveform, lengths)
            probs = F.softmax(logits_emotion, dim=-1)
            
        self.current_probs = probs.squeeze().cpu().numpy()
        majority_class_idx = torch.argmax(probs, dim=-1).item()
        self.current_emotion = EMOTION_MAP[majority_class_idx]

        print(f"[EMOTION] Classified emotion: {self.current_emotion} (Confidence: {self.current_probs[majority_class_idx]:.2f})")

        self.label_history.append((self.current_emotion, self.total_samples_seen))


    def update_plot(self, frame):
        if self.need_classify:
            self.classify_and_transcribe()
            self.need_classify = False

        # Emotion bar update
        for i, bar in enumerate(self.bar):
            bar.set_height(self.current_probs[i])
            if EMOTION_MAP[i] == self.current_emotion:
                bar.set_color('coral')
            else:
                bar.set_color('skyblue')
        self.ax_bar.set_title(f"Real-time Emotion Probabilities: {self.current_emotion}")


        # --- Waveform plot update logic ---
        current_data = self.buffer # Always work from the full buffer

        if self.use_envelope_plot:
            # Calculate rectified and smoothed envelope
            abs_data = np.abs(current_data)
            
            # Simple moving average for smoothing
            if self.envelope_smoothing_window > 0:
                kernel = np.ones(self.envelope_smoothing_window) / self.envelope_smoothing_window
                smoothed_data = np.convolve(abs_data, kernel, mode='valid')
            else:
                smoothed_data = abs_data 
            
            # --- NEW: Downsample the smoothed data for plotting ---
            plot_data = smoothed_data[::self.downsample_factor]
            # Adjust time axis to match the downsampled data points
            plot_x = np.linspace(0, self.asr_window_sec, len(plot_data))
            
            #self.ax_wave.set_ylim(0, 1.0) # Envelope is always positive
            self.wave_line.set_color('purple') # Color for envelope
            self.wave_line.set_linewidth(1.5) # Slightly thicker for envelope
        else:
            # Downsampled raw waveform
            plot_data = current_data[::self.downsample_factor]
            plot_x = np.linspace(0, self.asr_window_sec, len(plot_data))
            
            #self.ax_wave.set_ylim(-1.0, 1.0) # Raw waveform is bipolar
            self.wave_line.set_color('darkblue') # Color for raw waveform
            self.wave_line.set_linewidth(1) # Thinner for raw waveform

        self.wave_line.set_ydata(plot_data)
        self.wave_line.set_xdata(plot_x)
        
        # Update text annotations
        self.emotion_text_handle.set_text(f"Emotion: {self.current_emotion}")
        self.asr_text_handle.set_text(f"Transcript: [{self.current_text}]" if self.current_text else "ASR: No Speech")

        return self.bar, self.wave_line, self.emotion_text_handle, self.asr_text_handle

    def run(self):
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.step_size,
                device=self.device_audio
            ):
                print(f"[INFO] Starting audio stream. Recording at {self.sample_rate} Hz.")
                print(f"[INFO] Buffer size: {self.asr_window_sec} seconds. Classifying every {self.classify_interval_sec} seconds.")
                ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"[ERROR] An error occurred during audio stream: {e}")
            print("Please ensure your audio device is correctly selected and available.")
            print("You can list available devices using: python -m sounddevice")


if __name__ == "__main__":
    emotion_model, target_sr = load_model_from_config("configs/inference_config.yml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    realtime = RealTimeEmotionAndASR(emotion_model, target_sr, device=device)
    
    realtime.run()