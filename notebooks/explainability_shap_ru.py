# SHAP Explanation for MIMII Predictions (Python)

import shap
import torch
import numpy as np
from src.model import SoundAnomalyAutoencoder
from src.feature_extraction import FeatureExtractor

# Пример использования SHAP c автоэнкодером для объяснения аномального score
model = SoundAnomalyAutoencoder()
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()
feature_extractor = FeatureExtractor()

def explain_anomaly(audio_data):
    features = feature_extractor.extract_mel_spectrogram(audio_data)
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    def predict_fn(x):
        x = torch.FloatTensor(x).unsqueeze(1)  # п.с. B,1,128,T
        rec, _ = model(x)
        return ((x - rec) ** 2).mean(dim=[1,2,3]).detach().numpy()
    explainer = shap.Explainer(predict_fn, features.numpy(), feature_names=[f"mel_{i}" for i in range(features.shape[1])])
    shap_values = explainer(features.numpy())
    shap.summary_plot(shap_values, features.numpy())
# Использовать: передать сигнал из тестового файла для объяснения