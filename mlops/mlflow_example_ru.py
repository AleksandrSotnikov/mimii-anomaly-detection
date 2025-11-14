# MLflow Tracking Example (Python)

import mlflow
from src.model import SoundAnomalyAutoencoder
from src.evaluate import Evaluator

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MIMII Anomaly Detection")

with mlflow.start_run():
    # Логгирование гиперпараметров
    mlflow.log_param("model_type", "autoencoder")
    mlflow.log_param("latent_dim", 128)
    # Логгирование веса модели
    mlflow.pytorch.log_model(SoundAnomalyAutoencoder(), "model")
    # Логгирование метрик
    evaluator = Evaluator(SoundAnomalyAutoencoder(), device="cpu")
    metrics = {"auc_roc": 0.87, "precision": 0.91, "recall": 0.85}
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    mlflow.log_artifact("results/metrics.txt")
    # Для версии продвинутого сценария: mlflow.<add artifact/plot/logged image> (см. доки)
