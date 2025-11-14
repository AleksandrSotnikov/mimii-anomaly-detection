# Monitoring and Analytics Example (Prometheus + FastAPI)

from prometheus_client import start_http_server, Summary, Counter
from fastapi import FastAPI, UploadFile
from src import SoundAnomalyAutoencoder, FeatureExtractor
from src.utils import get_device
import torch, numpy as np, librosa
import logging

MODEL_LATENCY = Summary('mimii_model_inference_latency_seconds', 'Time spent on inference')
REQUESTS = Counter('mimii_requests_total', 'Total number of requests processed')
ANOMALIES = Counter('mimii_anomalies_total', 'Total number of anomalies detected')

app = FastAPI()
logger = logging.getLogger("api_monitoring")
device = get_device()
model = SoundAnomalyAutoencoder().to(device)
model.eval()
feature_extractor = FeatureExtractor()

@app.on_event('startup')
def init_metrics():
    start_http_server(8001)

@app.post("/predict_monitor/")
@MODEL_LATENCY.time()
async def predict_monitor(file: UploadFile):
    REQUESTS.inc()
    contents = await file.read()
    audio = np.frombuffer(contents, dtype=np.float32)
    sr = 16000
    audio = librosa.util.fix_length(audio, size=sr*5)
    features = feature_extractor.extract_mel_spectrogram(audio)
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed, latent = model(features)
        error = torch.mean((features - reconstructed) ** 2).item()
    if error > 0.7:  # пример динамического порога
        ANOMALIES.inc()
    return {"anomaly_score": error}
