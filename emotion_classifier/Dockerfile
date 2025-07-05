# Use NVIDIA PyTorch base image with CUDA 12.1
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime


# Set working directory inside the container
WORKDIR /app

# Copy only the deploy folder contents to avoid bringing in unnecessary training files
COPY ./deploy/ /app/

# Install Python dependencies with correct CUDA-index fallback
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 --extra-index-url https://download.pytorch.org/whl/cu121

 RUN apt-get update && apt-get install -y libsndfile1

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
