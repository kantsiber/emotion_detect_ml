import json
import torch
import torchaudio
import asyncio
import numpy as np
import uuid
import base64
import opensmile
import torchaudio.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from starlette.middleware.cors import CORSMiddleware
from sympy.printing.pytorch import torch
from pydantic import BaseModel
from models.EmoModel_base import EmoModel
from typing import Dict, List, Any


def load_config(config_path: str) -> dict:
    """Загрузка конфигурационного файла"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


# ==========================================
# Загрузка конфигурации
# ==========================================
config_path = "config.json"

print(f"Загрузка конфига из {config_path}")
config = load_config(config_path)

checkpoints_dir = config["check_points_path"]
num_classes = config["num_classes"]
target_sample_rate = config["target_sample_rate"]
target_time = config.get("target_time", 3.0)  # Динамическое значение из config

# Параметры спектрограммы
n_mels = 80
n_fft = 1024
hop_length = 256

# Маппинг эмоций
emotion_to_label = {
    'neutral': 0,
    'sad': 1,
    'angry': 2,
    'positive': 3
}

# Инвертированный маппинг для получения названия по индексу
label_to_emotion = {v: k for k, v in emotion_to_label.items()}

# ==========================================
# Инициализация модели
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = Path(checkpoints_dir) / "model_check_points"
epoch_num = 9
model_path = f"../checkpoints/base_model/model_check_points/check_point_{epoch_num}.pth"

model = EmoModel(num_classes=num_classes)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
model.eval()

print(f"Модель загружена на {device}")
print(f"target_time из конфига: {target_time} сек")

# ==========================================
# Инициализация трансформов
# ==========================================
mel_transform = T.MelSpectrogram(
    sample_rate=target_sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
)

# Инициализация openSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ==========================================
# FastAPI приложение
# ==========================================
app = FastAPI(
    title="Emotion Recognition API",
    version="2.0.0",
    description="API для распознавания эмоций в аудио с использованием CNN и openSMILE"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище задач
tasks_storage: Dict[str, Dict] = {}


class AudioUploadRequest(BaseModel):
    """Запрос с аудио в виде вектора"""
    audioData: List[float]  # Вектор аудио данных
    sampleRate: int  # Sample rate аудио

    class Config:
        json_schema_extra = {
            "example": {
                "audioData": [0.001, -0.002, 0.003],
                "sampleRate": 48000
            }
        }


class Segment(BaseModel):
    """Сегмент эмоции"""
    label: str
    startFrame: int
    endFrame: int


class FeaturesRow(BaseModel):
    """Строка таблицы признаков"""
    name: str
    values: List[float]


class FeaturesTable(BaseModel):
    """Таблица признаков"""
    columns: List[str]
    rows: List[FeaturesRow]


class SpectrogramData(BaseModel):
    """Данные спектрограммы"""
    type: str = "mel"
    frames: int
    melBins: int
    hopLength: int
    minDb: int
    maxDb: int
    format: str = "uint8"
    data: str


# ==========================================
# API Endpoints
# ==========================================
@app.post("/api/upload")
async def upload_audio(request: AudioUploadRequest):
    """
    Endpoint для загрузки аудио в виде вектора

    Принимает:
    - audioData: массив float значений (аудио сигнал)
    - sampleRate: частота дискретизации в Hz

    Возвращает:
    - taskId: ID задачи для отслеживания статуса
    - status: начальный статус (queued)
    """
    try:
        task_id = str(uuid.uuid4())

        # Конвертация списка в тензор
        waveform = torch.tensor(request.audioData, dtype=torch.float32)

        # Проверка размерности
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Добавляем канал [1, samples]

        print(f"\n{'=' * 60}")
        print(f"Новая задача: {task_id}")
        print(f"Получен аудио вектор: {waveform.shape}")
        print(f"Sample rate: {request.sampleRate} Hz")
        print(f"Длительность: {waveform.shape[1] / request.sampleRate:.2f} сек")
        print(f"{'=' * 60}")

        # Создание задачи
        tasks_storage[task_id] = {
            "taskId": task_id,
            "status": "queued",
            "progress": 0,
        }

        # Запускаем обработку в фоне
        asyncio.create_task(process_audio_tensor(task_id, waveform, request.sampleRate))

        return {
            "ok": True,
            "taskId": task_id,
            "status": "queued"
        }

    except Exception as e:
        print(f"\n{'!' * 60}")
        print(f"ОШИБКА загрузки аудио")
        print(f"{'!' * 60}")
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'!' * 60}\n")

        return {
            "ok": False,
            "error": str(e)
        }


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Получение статуса и результатов задачи

    Статусы:
    - queued: задача в очереди
    - processing: обработка идет
    - done: обработка завершена, результаты доступны
    - error: произошла ошибка
    """
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    # Убираем внутренние поля
    response = {k: v for k, v in task.items() if k not in ["waveform", "sample_rate"]}

    return response


# ==========================================
# Основная логика обработки
# ==========================================
async def process_audio_tensor(task_id: str, waveform: torch.Tensor, sample_rate: int):
    """
    Основная функция обработки аудио из тензора

    Этапы:
    1. Ресемплинг (если нужно)
    2. Создание спектрограммы
    3. Анализ по сегментам (эмоции)
    4. Подсчет распределения эмоций
    5. Извлечение openSMILE дескрипторов
    6. Подготовка данных для фронтенда
    """
    try:
        tasks_storage[task_id]["status"] = "processing"
        tasks_storage[task_id]["progress"] = 0.1

        print(f"\n{'=' * 60}")
        print(f"ОБРАБОТКА ЗАДАЧИ: {task_id}")
        print(f"{'=' * 60}")
        print(f"Исходный sample_rate: {sample_rate} Hz")
        print(f"Длительность: {waveform.shape[1] / sample_rate:.2f} сек")

        # ==========================================
        # Шаг 1: Ресемплинг
        # ==========================================
        if sample_rate != target_sample_rate:
            print(f"Ресемплинг: {sample_rate} Hz -> {target_sample_rate} Hz")
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        # ==========================================
        # Шаг 2: Создание полной спектрограммы
        # ==========================================
        full_mel_spec = mel_transform(waveform)
        full_mel_spec_db = torch.log(full_mel_spec + 1e-6)
        full_mel_spec_norm = (full_mel_spec_db - full_mel_spec_db.mean()) / (full_mel_spec_db.std() + 1e-6)

        print(f"Спектрограмма создана: {full_mel_spec_norm.shape}")
        tasks_storage[task_id]["progress"] = 0.3

        # ==========================================
        # Шаг 3: Анализ по сегментам
        # ==========================================
        target_length = int(target_sample_rate * target_time)
        segments = []
        emotion_predictions = []

        total_samples = waveform.shape[1]
        num_segments = max(1, total_samples // target_length)

        # Если аудио короче target_time, обрабатываем как 1 сегмент
        if total_samples < target_length:
            num_segments = 1

        print(f"\nАНАЛИЗ ЭМОЦИЙ:")
        print(f"Сегментов: {num_segments} (по {target_time} сек)")
        print("-" * 60)

        for i in range(num_segments):
            start_sample = i * target_length
            end_sample = min(start_sample + target_length, total_samples)
            segment_wave = waveform[:, start_sample:end_sample]

            # Padding если сегмент короче
            if segment_wave.shape[1] < target_length:
                padding = target_length - segment_wave.shape[1]
                segment_wave = torch.nn.functional.pad(segment_wave, (0, padding))

            # Спектрограмма сегмента
            segment_mel = mel_transform(segment_wave)
            segment_mel = torch.log(segment_mel + 1e-6)
            segment_mel = (segment_mel - segment_mel.mean()) / (segment_mel.std() + 1e-6)

            # Предсказание эмоции
            with torch.no_grad():
                segment_mel_input = segment_mel.unsqueeze(0).to(device)
                output = model(segment_mel_input)
                pred_idx = output.argmax(1).item()

                emotion = label_to_emotion[pred_idx]

                probs = torch.softmax(output, dim=1)
                confidence = probs[0, pred_idx].item()

                # Координаты в frames спектрограммы
                start_frame = start_sample // hop_length
                end_frame = end_sample // hop_length

                segments.append({
                    "label": emotion,
                    "startFrame": start_frame,
                    "endFrame": end_frame
                })

                emotion_predictions.append(emotion)

                print(f"Сегмент {i + 1}/{num_segments}: {emotion:>8} (уверенность: {confidence:.1%})")

        tasks_storage[task_id]["progress"] = 0.5

        # ==========================================
        # Шаг 4: Подсчет количества эмоций по сегментам
        # ==========================================
        emotion_counts = {}
        for emotion in emotion_predictions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Определение главной эмоции
        main_emotion = max(set(emotion_predictions), key=emotion_predictions.count)

        print(f"\n{'=' * 60}")
        print(f"ГЛАВНАЯ ЭМОЦИЯ: {main_emotion.upper()}")
        print(f"\nРаспределение эмоций:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotion_predictions)) * 100
            print(f"  {emotion:>8}: {count:>2} сегментов ({percentage:>5.1f}%)")
        print(f"{'=' * 60}")

        tasks_storage[task_id]["progress"] = 0.6

        # ==========================================
        # Шаг 5: Извлечение openSMILE дескрипторов
        # ==========================================
        print(f"\nИЗВЛЕЧЕНИЕ OPENSMILE ПРИЗНАКОВ...")

        # Конвертация в numpy для openSMILE
        waveform_np = waveform.squeeze(0).numpy()

        # Извлечение признаков
        features_df = smile.process_signal(waveform_np, sample_rate)

        # Конвертация DataFrame в словарь дескрипторов
        features_data_descriptor = {}
        for column in features_df.columns:
            value = features_df[column].iloc[0]
            features_data_descriptor[column] = float(value)

        print(f"OpenSMILE дескрипторов извлечено: {len(features_data_descriptor)}")

        # Вывод примеров дескрипторов
        print(f"\nПримеры дескрипторов:")
        example_features = list(features_data_descriptor.items())[:5]
        for name, value in example_features:
            print(f"  {name}: {value:.4f}")

        tasks_storage[task_id]["progress"] = 0.8

        # ==========================================
        # Шаг 6: Подготовка спектрограммы для фронтенда
        # ==========================================
        mel_for_display = full_mel_spec_norm.squeeze(0).transpose(0, 1).numpy()

        mel_min = mel_for_display.min()
        mel_max = mel_for_display.max()

        if mel_max > mel_min:
            mel_uint8 = ((mel_for_display - mel_min) / (mel_max - mel_min) * 255).astype(np.uint8)
        else:
            mel_uint8 = np.zeros_like(mel_for_display, dtype=np.uint8)

        mel_base64 = numpy_to_base64(mel_uint8)

        frames, mel_bins = mel_uint8.shape
        print(f"\nСпектрограмма для фронтенда: {frames} frames × {mel_bins} mel bins")

        tasks_storage[task_id]["progress"] = 0.9

        # ==========================================
        # Шаг 7: Формирование финального результата
        # ==========================================
        tasks_storage[task_id].update({
            "status": "done",
            "progress": 1.0,

            "summary": {
                "mainEmotion": main_emotion,
                "emotionCounts": emotion_counts,  # Словарь с подсчетом эмоций
            },

            "segments": segments,

            # Табличное представление для UI
            "features": format_opensmile_features(features_data_descriptor),

            # Полный словарь дескрипторов для прямого доступа
            "featuresDataDescriptor": features_data_descriptor,

            "spectrogram": {
                "type": "mel",
                "frames": frames,
                "melBins": mel_bins,
                "hopLength": hop_length,
                "minDb": -80,
                "maxDb": 0,
                "format": "uint8",
                "data": mel_base64
            }
        })

        print(f"\n{'=' * 60}")
        print(f"ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f"{'=' * 60}")
        print(f"Task ID: {task_id}")
        print(f"Главная эмоция: {main_emotion}")
        print(f"Всего сегментов: {len(segments)}")
        print(f"Дескрипторов: {len(features_data_descriptor)}")
        print(f"Спектрограмма: {frames}×{mel_bins}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n{'!' * 60}")
        print(f"ОШИБКА ОБРАБОТКИ ЗАДАЧИ {task_id}")
        print(f"{'!' * 60}")
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'!' * 60}\n")

        tasks_storage[task_id]["status"] = "error"
        tasks_storage[task_id]["error"] = str(e)
        tasks_storage[task_id]["progress"] = 0


# ==========================================
# Вспомогательные функции
# ==========================================
def numpy_to_base64(array: np.ndarray) -> str:
    """
    Конвертация numpy массива в base64 строку

    Args:
        array: numpy массив (обычно uint8)

    Returns:
        base64 строка
    """
    flat_array = array.flatten()
    array_bytes = flat_array.tobytes()
    base64_str = base64.b64encode(array_bytes).decode('utf-8')
    return base64_str


def format_opensmile_features(features_dict: Dict[str, float]) -> Dict:
    """
    Форматирование openSMILE признаков для фронтенда
    Согласно контракту: { columns: [...], rows: [...] }

    Args:
        features_dict: словарь {имя_признака: значение}

    Returns:
        Табличное представление для UI
    """
    if not features_dict:
        return {"columns": ["value"], "rows": []}

    rows = []
    for feature_name, value in features_dict.items():
        row = {
            "name": feature_name,
            "values": [value]
        }
        rows.append(row)

    return {
        "columns": ["value"],
        "rows": rows
    }


# ==========================================
# Служебные endpoints
# ==========================================
@app.get("/health")
async def health_check():
    """
    Проверка состояния сервера

    Возвращает информацию о:
    - Загруженности модели
    - Устройстве (CPU/GPU)
    - Конфигурации
    - Количестве активных задач
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "tasks_count": len(tasks_storage),
        "tasks_active": len([t for t in tasks_storage.values() if t["status"] in ["queued", "processing"]]),
        "tasks_completed": len([t for t in tasks_storage.values() if t["status"] == "done"]),
        "tasks_failed": len([t for t in tasks_storage.values() if t["status"] == "error"]),
        "config": {
            "num_classes": num_classes,
            "sample_rate": target_sample_rate,
            "target_time": target_time,
            "emotions": list(emotion_to_label.keys()),
            "checkpoint": str(Path(model_path).name),
            "opensmile_feature_set": "ComParE_2016",
            "opensmile_features_count": 6373
        }
    }


@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Emotion Recognition API",
        "model": "EmoModel (CNN)",
        "version": "2.0.0",
        "emotions": list(emotion_to_label.keys()),
        "endpoints": {
            "upload": "POST /api/upload - загрузка аудио вектора",
            "task_status": "GET /api/task/{task_id} - статус задачи",
            "health": "GET /health - проверка состояния сервера",
            "docs": "GET /docs - Swagger документация"
        },
        "input_format": {
            "audioData": "array of floats (audio samples)",
            "sampleRate": "integer (Hz)"
        },
        "output_fields": {
            "summary": {
                "mainEmotion": "string - основная эмоция",
                "emotionCounts": "dict - количество каждой эмоции"
            },
            "segments": "array - список сегментов с эмоциями",
            "features": "table - табличное представление признаков",
            "featuresDataDescriptor": "dict - полный словарь openSMILE дескрипторов",
            "spectrogram": "object - данные спектрограммы в base64"
        }
    }


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """
    Удаление задачи из хранилища
    Полезно для очистки памяти
    """
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    del tasks_storage[task_id]
    return {"ok": True, "message": f"Task {task_id} deleted"}


@app.get("/api/tasks")
async def list_tasks():
    """
    Список всех задач
    Полезно для отладки
    """
    return {
        "total": len(tasks_storage),
        "tasks": [
            {
                "taskId": task["taskId"],
                "status": task["status"],
                "progress": task.get("progress", 0)
            }
            for task in tasks_storage.values()
        ]
    }


# ==========================================
# Запуск сервера
# ==========================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("EMOTION RECOGNITION API v2.0")
    print("=" * 70)
    print(f"Модель: EmoModel (CNN)")
    print(f"Классов: {num_classes}")
    print(f"Эмоции: {', '.join(emotion_to_label.keys())}")
    print(f"Устройство: {device}")
    print(f"Чекпоинт: {model_path}")
    print(f"Target time: {target_time} сек (из конфига)")
    print(f"Sample rate: {target_sample_rate} Hz")
    print(f"OpenSMILE: ComParE_2016 (6373 признака)")
    print("=" * 70)
    print(f"API адрес: http://localhost:8080")
    print(f"Документация: http://localhost:8080/docs")
    print(f"Health check: http://localhost:8080/health")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")




