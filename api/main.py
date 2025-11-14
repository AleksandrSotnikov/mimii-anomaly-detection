# FastAPI REST API for MIMII

from fastapi import FastAPI, File, UploadFile
from src import SoundAnomalyAutoencoder, FeatureExtractor
from src.utils import get_device, set_seed
import torch
import numpy as np
import librosa
import io
import logging

app = FastAPI()
logger = logging.getLogger("api")

# Load model
device = get_device()
model = SoundAnomalyAutoencoder().to(device)
model.eval()
feature_extractor = FeatureExtractor()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    audio = np.frombuffer(contents, dtype=np.float32)
    sr = 16000
    audio = librosa.util.fix_length(audio, size=sr*5) # 5 sec segment
    features = feature_extractor.extract_mel_spectrogram(audio)
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device) # Shape (1, 1, 128, T)
    with torch.no_grad():
        reconstructed, latent = model(features)
        error = torch.mean((features - reconstructed) ** 2).item()
    return {"anomaly_score": error}
