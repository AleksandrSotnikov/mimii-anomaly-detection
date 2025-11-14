# Transfer Learning Example for MIMII (PyTorch)

from src import MIMIIDataLoader, FeatureExtractor, SoundAnomalyAutoencoder, Trainer
from src.utils import get_device, set_seed

# Опция: дообучить автоэнкодер, предобученный на fans → pumps

set_seed(123)
device = get_device()

# 1. Протяжка кодировщика с fan на pumps
autoencoder_fan = SoundAnomalyAutoencoder().to(device)
# load weights from a pre-trained fan model
autoencoder_fan.load_state_dict(torch.load('checkpoints/fan_id00_best.pth')['model_state_dict'])

# 2. Новый загрузчик на pumps
data_loader = MIMIIDataLoader('data/', machine_type='pump', model_id='00', db_level=6)
feature_extractor = FeatureExtractor()
train_loader, val_loader, test_loader = data_loader.get_dataloaders(
    batch_size=32, feature_extractor=feature_extractor)

# 3. Дообучение с сохранением латентных слоев
def freeze_encoder(model):
    for n, p in model.encoder.named_parameters():
        p.requires_grad = False
freeze_encoder(autoencoder_fan)

trainer = Trainer(autoencoder_fan, device=device, learning_rate=5e-4)
history = trainer.train(train_loader, val_loader, num_epochs=50, patience=10)

# 4. Оценка эффективности трансфера
from src import Evaluator
evaluator = Evaluator(autoencoder_fan, device=device)
metrics = evaluator.evaluate(test_loader)
print(f'Transfer AUC-ROC: {metrics["auc_roc"]:.4f}')
