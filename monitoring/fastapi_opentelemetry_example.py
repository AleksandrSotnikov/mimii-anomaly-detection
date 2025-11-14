# OpenTelemetry example for APM tracing (FastAPI)

from fastapi import FastAPI, UploadFile
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src import SoundAnomalyAutoencoder, FeatureExtractor
from src.utils import get_device
import torch, numpy as np, librosa
import logging

app = FastAPI()
device = get_device()
model = SoundAnomalyAutoencoder().to(device)
model.eval()
feature_extractor = FeatureExtractor()

# Setup OpenTelemetry Tracing
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
FastAPIInstrumentor.instrument_app(app)

@app.post("/predict_trace/")
async def predict_trace(file: UploadFile):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("predict"):
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
