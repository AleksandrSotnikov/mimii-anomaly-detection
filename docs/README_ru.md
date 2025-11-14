# MIMII Anomaly Detection — Документация

## Оглавление
- [Введение](#введение)
- [Архитектура проекта](#архитектура-проекта)
- [Установка и запуск](#установка-и-запуск)
- [Работа с датасетом](#работа-с-датасетом)
- [Примеры кода](#примеры-кода)
- [Запуск обучения и оценки](#запуск-обучения-и-оценки)
- [API-интерфейс (FastAPI)](#api-интерфейс-fastapi)
- [Отладка и тестирование](#отладка-и-тестирование)
- [Разработка и вклад](#разработка-и-вклад)

---

## Введение
MIMII — открытая платформа для обнаружения аномалий на звуковых сигналах промышленного оборудования с помощью нейросетей. Является бенчмарком для задач машинного обучения без учителя, устойчивого к шуму и адаптивного к новым типам неисправностей.

## Архитектура проекта
Проект построен по модульному принципу:
- src/data_loader.py – загрузка и препроцессинг
- src/feature_extraction.py – преобразование аудио
- src/model.py – нейросетевые архитектуры (Autoencoder, VAE)
- src/train.py – цикл обучения
- src/evaluate.py – оценка и визуализация
- src/utils.py — логгирование, config, reproducibility

## Установка и запуск
```bash
git clone https://github.com/AleksandrSotnikov/mimii-anomaly-detection.git
cd mimii-anomaly-detection
pip install -r requirements.txt
```

## Работа с датасетом
Скачайте архивы с [Zenodo](https://zenodo.org/records/3384388), распакуйте в папку data/. Структура:
```
data/
└─ 6dB_fan/
   └─ id_00/
      └─ normal_001.wav
      └─ anomaly_003.wav
```

## Примеры кода
**Загрузка и препроцессинг:**
```python
from src import MIMIIDataLoader, FeatureExtractor
loader = MIMIIDataLoader('data/', machine_type='fan', model_id='00')
feature_extractor = FeatureExtractor()
train_loader, val_loader, test_loader = loader.get_dataloaders(
    batch_size=32,
    feature_extractor=feature_extractor
)
```

**Обучение модели:**
```python
from src import SoundAnomalyAutoencoder, Trainer
model = SoundAnomalyAutoencoder().to('cuda')
trainer = Trainer(model, device='cuda', learning_rate=1e-3)
history = trainer.train(
    train_loader, val_loader,
    num_epochs=100, patience=20
)
```

**Оценка результатов и визуализация:**
```python
from src import Evaluator
evaluator = Evaluator(model, device='cuda')
metrics = evaluator.evaluate(test_loader)
evaluator.generate_report(metrics, 'results/')
```

## Запуск обучения и оценки через CLI
```bash
python train_main.py --data-dir data/ --machine-type fan --model-id 00 --batch-size 32 --epochs 100
python evaluate_main.py --checkpoint checkpoints/best_model.pth --data-dir data/ --machine-type fan --model-id 00
```

## API-интерфейс (FastAPI)
Ветка содержит пример REST API для инференса, загрузки файлов и управления моделями:
```python
from fastapi import FastAPI, UploadFile
from src import SoundAnomalyAutoencoder, FeatureExtractor
# ...
app = FastAPI()
@app.post('/predict/')
async def predict(file: UploadFile):
    audio = await file.read()
    # preprocess audio, extract features, run model, return anomaly score
```
### Пример запуска:
```bash
uvicorn api.main:app --reload --port 8080
```

## Отладка и тесты
- Запуск unit-тестов: `pytest tests/`
- Лог файлы доступны в logs/
- Внутренние ошибки логируются в train.log

## Разработка и вклад
- Все важные задачи оформляются через Issues.
- Весь новый функционал реализуется в отдельных ветках через pull request.
- Автоматические проверки (CI/CD pipeline) планируются для Pytest, Pylint, Black.
