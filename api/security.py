# REST API Security — Rate limiting, basic auth

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import time
import secrets
from src import SoundAnomalyAutoencoder, FeatureExtractor
from src.utils import get_device
import torch
import numpy as np
import librosa
import logging

RATE_LIMIT = 10 # запросов в минуту на IP
API_USERS = {
    'admin': 'secretpass'
}
rate_limiter = {}
security = HTTPBasic()

app = FastAPI()
logger = logging.getLogger("api_security")
device = get_device()
model = SoundAnomalyAutoencoder().to(device)
model.eval()
feature_extractor = FeatureExtractor()

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, 'admin')
    correct_password = secrets.compare_digest(credentials.password, API_USERS['admin'])
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

@app.middleware("http")
async def add_rate_limit(request: Request, call_next):
    ip = request.client.host
    now = time.time()
    count_ts = rate_limiter.get(ip, [])
    # clean expired
    count_ts = [ts for ts in count_ts if now - ts < 60]
    if len(count_ts) >= RATE_LIMIT:
        return HTTPException(status_code=429, detail="Too Many Requests")
    count_ts.append(now)
    rate_limiter[ip] = count_ts
    response = await call_next(request)
    return response

@app.post("/predict_secure/")
async def predict_secure(request: Request, credentials: HTTPBasicCredentials = Depends(check_auth)):
    post = await request.form()
    file = post['file']
    contents = await file.read()
    audio = np.frombuffer(contents, dtype=np.float32)
    sr = 16000
    audio = librosa.util.fix_length(audio, size=sr*5)
    features = feature_extractor.extract_mel_spectrogram(audio)
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed, latent = model(features)
        error = torch.mean((features - reconstructed) ** 2).item()
    return {"anomaly_score": error}
