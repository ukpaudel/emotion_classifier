# Emotion and ASR API

This repository contains a FastAPI application that provides an API for emotion recognition and Automatic Speech Recognition (ASR) from audio inputs. The application leverages pre-trained models, specifically a HuBERT-based model for emotion classification and a Wav2Vec2 model for ASR.

The project is designed to be run within a Docker container to ensure a consistent environment and leverage GPU acceleration.

## Table of Contents

1.  [Prerequisites](#1-prerequisites)
2.  [Project Structure](#2-project-structure)
3.  [Setup Instructions](#3-setup-instructions)
    * [3.1. Clone the Repository](#31-clone-the-repository)
    * [3.2. Docker Image Build](#32-docker-image-build)
    * [3.3. Run the Docker Container](#33-run-the-docker-container)
4.  [API Usage](#4-api-usage)
5.  [Troubleshooting](#5-troubleshooting)

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Git:** For cloning the repository.
    * [Download Git](https://git-scm.com/downloads)
* **Docker Desktop:** This includes Docker Engine and Docker Compose, essential for building and running Docker containers.
    * [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
* **NVIDIA GPU (Recommended):** A compatible NVIDIA GPU is highly recommended for accelerated inference.
    * Ensure you have the correct NVIDIA drivers installed that are compatible with **CUDA 12.4**. You can check your CUDA version by running `nvidia-smi` in your terminal.
    * **Note:** The Docker image uses PyTorch with CUDA 12.4.

## 2. Project Structure

The core application code and Dockerfile are organized as follows (paths relative to the repository root):

emotion_classifier/
├── Dockerfile                  # Defines how to build the Docker image
├── emotion_classifier/         # Your main application source folder
│   └── deploy/                 # Contains the FastAPI app, models, config, and requirements
│       ├── app.py              # FastAPI application entry point
│       ├── models/             # Contains model-related classes
│       │   ├── emotion_model.py
│       │   └── attention_classifier.py
│       ├── inference_config.yml# Configuration for model loading
│       ├── hubert_2MLP_0Enc_noisedata_aug_cosinewrmst_D0p3_v2_file19.pt # Your pre-trained model weights
│       ├── requirements.txt    # Python dependencies for the application
│       ├── static/             # Static files for the web interface
│       │   └── style.css
│       └── templates/          # HTML templates for the web interface
│           └── index.html
└── README.md                   # This file


## 3. Setup Instructions

Follow these steps to get the API running in a Docker container on your local machine.

### 3.1. Clone the Repository

First, clone this Git repository to your local machine:

```bash
git clone [https://github.com/ukpaudel/emotion_classifier.git](https://github.com/ukpaudel/emotion_classifier.git) -b testbranch
cd emotion_classifier
Note: The -b testbranch flag specifies that you're cloning the testbranch directly.

3.2. Docker Image Build
Navigate to the root directory of the cloned repository (the emotion_classifier folder where the Dockerfile is located). Then, build the Docker image. This process will download the necessary base image, copy your application code, and install all Python dependencies.

Important: This step can take a significant amount of time the first time you run it, as it needs to download large base images and install many Python packages. Subsequent builds will be faster due to Docker's caching.

Bash

docker build -t emotion-asr-api .
docker build: The command to build a Docker image.

-t emotion-asr-api: Tags the image with the name emotion-asr-api.

.: Specifies that the build context (the files Docker can access) is the current directory (the repository root).

If the build completes successfully, you will see a message similar to Successfully built <image_id> and Successfully tagged emotion-asr-api:latest.

3.3. Run the Docker Container
Once the Docker image is built, you can run a container from it. This will start your FastAPI application.

Bash

docker run -p 8000:8000 --gpus all emotion-asr-api
docker run: Creates and starts a new container from an image.

-p 8000:8000: Maps port 8000 on your host machine to port 8000 inside the container. This is how you'll access the API from your browser or other applications.

--gpus all: Crucial for GPU acceleration. This flag tells Docker to provide access to all available NVIDIA GPUs on your host machine to the container. If you don't have an NVIDIA GPU or the drivers aren't set up, you might omit this flag (but inference will be much slower on CPU).

emotion-asr-api: The name of the Docker image you just built.

After running this command, you should see output from Uvicorn (FastAPI's server) in your terminal, indicating that the application has started and is listening on 0.0.0.0:8000.

4. API Usage
Once the container is running:

Open your web browser.

Navigate to http://localhost:8000. This will load the main web interface for the application.

On the web page, you should see an option to upload a short .wav audio file (a few seconds long). Select your audio file and submit it to receive emotion and ASR predictions.

Alternatively, you can access the interactive API documentation (Swagger UI) by navigating to http://localhost:8000/docs. Here, you can manually test the API endpoints.

5. Troubleshooting
FROM requires either one or three arguments: This error usually means there's an inline comment (#) on the same line as the FROM or COPY instruction in your Dockerfile. Ensure these lines are clean, e.g., FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime and COPY emotion_classifier/deploy/ /app/.

"/emotion_classifier/deploy": not found: This means the COPY instruction in your Dockerfile can't find the source directory.

Ensure your Dockerfile is in the outermost emotion_classifier folder.

Ensure the COPY line correctly spells out the path, e.g., COPY emotion_classifier/deploy/ /app/. The path is relative to the Dockerfile's location.

ModuleNotFoundError or NameError for Wav2Vec2Processor, Wav2Vec2ForCTC, etc.:

Ensure these classes are correctly imported at the top of your app.py (e.g., from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC).

Verify that sentencepiece is listed in your requirements.txt.

After making changes to Python code (.py files) or requirements.txt, you must rebuild the Docker image (docker build -t emotion-asr-api .) and then re-run the container.

TypeError: 'Wav2Vec2Bundle' object is not subscriptable or AttributeError: 'Wav2Vec2Bundle' object has no attribute 'feature_dim': This means you're trying to access parts of the torchaudio.pipelines.HUBERT_BASE (or similar bundle) incorrectly.

Change encoder_bundle["model"] to encoder_bundle.get_model().

Change encoder_bundle["sample_rate"] to encoder_bundle.sample_rate.

Change encoder_bundle["feature_dim"] to self.encoder.feature_extractor.config.output_dim (assuming self.encoder is the model obtained via get_model()).

Remember to rebuild and re-run after fixing these.

Docker Daemon Not Running / Cannot Connect to Docker Daemon: Ensure Docker Desktop is running on your machine.

CUDA/GPU Issues: If --gpus all doesn't work, ensure your NVIDIA drivers are up-to-date and compatible with CUDA 12.4, and that Docker Desktop is configured to use the WSL 2 backend (Windows) or has proper GPU passthrough set up (Linux). Check Docker Desktop settings for GPU.