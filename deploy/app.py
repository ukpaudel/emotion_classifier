from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import torch
import torchaudio
import io
import numpy as np
import torch.nn.functional as F
import yaml
import os
import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC # Add this line
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# IMPORTANT: Adjust sys.path to allow importing from the copied 'models' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now, directly import your EmotionModel class
try:
    from models.emotion_model import EmotionModel # Adjust based on your actual module/class name
except ImportError as e:
    raise ImportError(f"Could not import EmotionModel. Ensure 'models' folder is copied to 'deploy/' and structure is correct. Error: {e}")


# --- Configuration Constants (SIMPLIFIED) ---
FIXED_AUDIO_WINDOW_SECONDS = 5.0 # Still relevant for Emotion Model's fixed window
ASR_MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self" # Directly from Hugging Face Hub

# --- Helper Functions ---

# load_model_from_config (No change, uses EmotionModel directly)
def load_model_from_config(config_path):
    """Loads emotion classification model from config without resolve_class."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = EmotionModel(**config['model']['args']) 
    
    checkpoint_path = config['model']['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, config['model']['sample_rate']

# ASR Model Loading Function (No change)
def load_asr_model():
    """Loads Wav2Vec2 ASR model and processor directly from Hugging Face."""
    print(f"Downloading/Loading ASR processor: {ASR_MODEL_NAME}")
    asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    print(f"Downloading/Loading ASR model: {ASR_MODEL_NAME}")
    asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME)
    asr_model.eval()
    return asr_model, asr_processor

# --- Core Inference Logic (SIMPLIFIED: Removed silence detection) ---
def process_audio_segment(
    audio_segment_np: np.ndarray, 
    emotion_model: torch.nn.Module, 
    asr_model: torch.nn.Module, 
    asr_processor: Wav2Vec2Processor, 
    sample_rate: int, 
    device: torch.device,
    emotion_map: dict,
    fixed_audio_window_samples: int # This is still needed for the emotion model's input size
):
    """
    Processes a single audio segment for ASR and emotion classification.
    Expects audio_segment_np to be a NumPy array, already normalized.
    """
    # Removed: buffer_rms calculation and silence detection logic

    # ASR Transcription
    asr_inputs = asr_processor(audio_segment_np, return_tensors="pt", sampling_rate=sample_rate)
    # Move everything in the batch to the target device
    asr_inputs = {k: v.to(device) for k, v in asr_inputs.items()}

    with torch.no_grad():
        asr_logits = asr_model(**asr_inputs).logits

    asr_pred_ids = torch.argmax(asr_logits, dim=-1)
    transcription = asr_processor.decode(asr_pred_ids[0]).strip()

    # Emotion Classification
    # Note: emotion_waveform uses fixed_audio_window_samples for its length matching
    emotion_waveform = torch.from_numpy(audio_segment_np).unsqueeze(0).to(device)
    emotion_lengths = torch.tensor([fixed_audio_window_samples]).to(device) # Still needed for model's forward pass
    
    with torch.no_grad():
        emotion_output = emotion_model(emotion_waveform, emotion_lengths)
        emotion_probs = F.softmax(emotion_output, dim=-1).squeeze().cpu().numpy()
        predicted_emotion_idx = np.argmax(emotion_probs)
        predicted_emotion = emotion_map[predicted_emotion_idx]

    return transcription, predicted_emotion, emotion_probs.tolist()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Speech Emotion and ASR Inference API",
    description="API to classify emotion and transcribe speech from audio files.",
    version="1.0.0"
)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Load Models Globally (Done once at API startup) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on device: {DEVICE}")

EMOTION_CONFIG_PATH = "./inference_config.yml"

EMOTION_MODEL, TARGET_SAMPLE_RATE = load_model_from_config(EMOTION_CONFIG_PATH)
EMOTION_MODEL.to(DEVICE)

ASR_MODEL, ASR_PROCESSOR = load_asr_model()
ASR_MODEL.to(DEVICE)

EMOTION_MAP = {
    0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad',
    4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'
}

# FIXED_AUDIO_WINDOW_SAMPLES is still crucial because your emotion model expects a 5-second input
FIXED_AUDIO_WINDOW_SAMPLES = int(FIXED_AUDIO_WINDOW_SECONDS * TARGET_SAMPLE_RATE)

# --- API Endpoint (No change, as it calls process_audio_segment) ---
@app.post("/inference", summary="Perform ASR and Emotion Classification on an audio file")
async def inference(audio: UploadFile = File(..., description="Audio file to process (WAV format recommended)")):
    """
    Accepts an audio file, processes it for ASR transcription and emotion classification.
    The audio will be resampled to the model's target sample rate and processed
    in a fixed 5-second window.
    """
    try:
        audio_bytes = await audio.read()
        if not audio.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV files are supported.")
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), format="wav")

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) 

        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE).to(DEVICE)
            waveform = resampler(waveform.to(DEVICE)).cpu() 

        if waveform.max().item() > 1.0 or waveform.min().item() < -1.0:
            max_abs_val = torch.max(torch.abs(waveform))
            if max_abs_val > 0:
                waveform = waveform / max_abs_val
        
        audio_np = waveform.squeeze(0).numpy() 

        # This fixed window logic is still necessary because your emotion model was trained on 5-second chunks.
        # It ensures that input audio is truncated or padded to 5 seconds.
        if len(audio_np) > FIXED_AUDIO_WINDOW_SAMPLES:
            processed_audio_segment_np = audio_np[:FIXED_AUDIO_WINDOW_SAMPLES]
        elif len(audio_np) < FIXED_AUDIO_WINDOW_SAMPLES:
            padding_needed = FIXED_AUDIO_WINDOW_SAMPLES - len(audio_np)
            processed_audio_segment_np = np.pad(audio_np, (0, padding_needed), 'constant', constant_values=0)
        else:
            processed_audio_segment_np = audio_np

        transcription, emotion, emotion_probs = process_audio_segment(
            processed_audio_segment_np, 
            EMOTION_MODEL, 
            ASR_MODEL, 
            ASR_PROCESSOR, 
            TARGET_SAMPLE_RATE, 
            DEVICE,
            EMOTION_MAP,
            FIXED_AUDIO_WINDOW_SAMPLES
        )

        return {
            "text": transcription.strip(),
            "emotion": emotion.capitalize(),
            "emotion_probs": [round(float(p), 4) for p in emotion_probs]
        }

    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")