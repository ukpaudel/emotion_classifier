
# Emotion and ASR API

This repository contains a FastAPI-based API for **emotion recognition** and **automatic speech recognition (ASR)** from audio inputs. It uses pre-trained models: a HuBERT-based model for emotion classification and a Wav2Vec2 model for ASR.

The project is designed to run in a Docker container for a consistent, GPU-accelerated environment.

---

## Table of Contents

- [Emotion and ASR API](#emotion-and-asr-api)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Project Structure](#project-structure)
  - [Setup Instructions](#setup-instructions)
    - [3.1 Clone the Repository](#31-clone-the-repository)
    - [3.2 Build the Docker Image](#32-build-the-docker-image)
    - [3.3 Run the Docker Container](#33-run-the-docker-container)
  - [API Usage](#api-usage)
    - [1️⃣ Web Interface](#1️⃣-web-interface)
    - [2️⃣ Swagger (Interactive API Docs)](#2️⃣-swagger-interactive-api-docs)
    - [3️⃣ Programmatic (cURL)](#3️⃣-programmatic-curl)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

---

## Prerequisites

Before you begin, ensure the following are installed:

- **Git** (for cloning the repository): [Download Git](https://git-scm.com/downloads)
- **Docker Desktop** (includes Docker Engine and Docker Compose): [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **NVIDIA GPU (recommended)** for accelerated inference.

> **Note**:
> - Ensure you have NVIDIA drivers compatible with **CUDA 12.4**.
> - Verify your CUDA version with `nvidia-smi` in the terminal.
> - The provided Docker image uses PyTorch with CUDA 12.4 support.

---

## Project Structure

```plaintext
emotion_classifier/
├── Dockerfile                # Defines the Docker image build
├── emotion_classifier/
│   └── deploy/               # Application source
│       ├── app.py            # FastAPI app entry point
│       ├── models/           # Model-related classes
│       │   ├── emotion_model.py
│       │   └── attention_classifier.py
│       ├── inference_config.yml
│       ├── hubert_2MLP_0Enc_noisedata_aug_cosinewrmst_D0p3_v2_file19.pt  # Pre-trained weights
│       ├── requirements.txt  # Python dependencies
│       ├── static/
│       │   └── style.css     # Web interface styling
│       └── templates/
│           └── index.html    # Web interface template
└── README.md                 # This file
```

---

## Setup Instructions

### 3.1 Clone the Repository

```bash
git clone https://github.com/ukpaudel/emotion_classifier.git -b testbranch
cd emotion_classifier
```

> The `-b testbranch` flag checks out the specific branch directly.

### 3.2 Build the Docker Image

```bash
docker build -t emotion-asr-api .
```

### 3.3 Run the Docker Container

```bash
docker run -p 8000:8000 --gpus all emotion-asr-api
```

---

## API Usage

### 1️⃣ Web Interface

- Go to [http://localhost:8000](http://localhost:8000)
- Upload a `.wav` file (3–4 seconds, recorded by a single speaker)
- Click **Upload & Analyze** to receive transcription and emotion predictions

### 2️⃣ Swagger (Interactive API Docs)

- Go to [http://localhost:8000/docs](http://localhost:8000/docs)
- Use the `POST /analyze` endpoint to upload audio and view JSON response

### 3️⃣ Programmatic (cURL)

```bash
curl -X POST "http://localhost:8000/analyze" -F "file=@path/to/your/audio.wav"
```

Sample response:

```json
{
  "transcription": "WE HAD A GREAT WEEKEND GOING TO LEGOLAND THIS WEEKEND",
  "predicted_emotion": "Angry",
  "emotion_scores": {
    "Neutral": 0.02,
    "Calm": 0.01,
    "Happy": 0.01,
    "Sad": 0.01,
    "Angry": 0.92,
    "Fearful": 0.01,
    "Disgust": 0.01,
    "Surprised": 0.01
  }
}
```

---

## Troubleshooting

- FROM/COPY line errors: remove inline comments from these lines
- Directory not found errors: check COPY paths relative to Dockerfile
- ModuleNotFoundError: check imports and `requirements.txt`, rebuild Docker image
- TypeError/AttributeError with bundles: use proper attribute/method access, rebuild
- Docker Daemon: ensure Docker Desktop is running
- GPU issues: check drivers, CUDA version, and Docker GPU config

---

## License

This project is licensed under your preferred license — *please add a LICENSE file if applicable.*
