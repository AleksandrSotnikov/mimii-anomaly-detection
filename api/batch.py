# Batch REST API endpoint for uploading multiple files/urls

from fastapi import FastAPI, File, UploadFile, Form
from typing import List
from src import SoundAnomalyAutoencoder, FeatureExtractor
from src.utils import get_device
import torch
import numpy as np
import librosa
import logging

app = FastAPI()
logger = logging.getLogger("api")

device = get_device()
model = SoundAnomalyAutoencoder().to(device)
model.eval()
feature_extractor = FeatureExtractor()

@app.post("/predict_batch/")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        audio = np.frombuffer(contents, dtype=np.float32)
        sr = 16000
        audio = librosa.util.fix_length(audio, size=sr*5)
        features = feature_extractor.extract_mel_spectrogram(audio)
        features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed, latent = model(features)
            error = torch.mean((features - reconstructed) ** 2).item()
        results.append({"name": file.filename, "anomaly_score": error})
    return {"results": results}
