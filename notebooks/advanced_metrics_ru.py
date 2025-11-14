# Advanced metrics: Precision/Recall curve, latency, memory profile, anomaly detection time
import torch, time, numpy as np
from src.evaluate import Evaluator
from src.model import SoundAnomalyAutoencoder
from sklearn.metrics import precision_recall_curve, auc

model = SoundAnomalyAutoencoder()
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

def measure_latency(test_loader, device='cpu'):
    times = []
    for batch, _ in test_loader:
        batch = batch.to(device)
        start = time.perf_counter()
        _ = model.reconstruction_error(batch)
        times.append(time.perf_counter()-start)
    return np.mean(times), np.std(times)

# Precision-Recall Curve
from src import Evaluator
pr_curve = precision_recall_curve

def plot_pr_curve(labels, scores):
    prec, rec, thr = pr_curve(labels, scores)
    pr_auc = auc(rec, prec)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rec, prec, label=f'AUC={pr_auc:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.grid(); plt.legend()
    plt.savefig('pr_curve.png')
    plt.close()

# Вызовы: analyzer.get_reconstruction_errors -> plot_pr_curve(labels, errors)