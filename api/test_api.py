import json
import subprocess
import torch
import asyncio
import numpy as np
import uuid
import base64
import opensmile
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pathlib import Path
from fastapi import Response
from fastapi.responses import FileResponse

from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.EmoModel_base import EmoModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import tempfile
import os
import random


# =========================================================
# Config
# =========================================================
def load_config(config_path: str) -> dict:
    """Загрузка конфигурационного файла"""
    with open(config_path, "r") as f:
        return json.load(f)


config_path = "config.json"
print(f"Загрузка конфига из {config_path}")
config = load_config(config_path)

checkpoints_dir = config["check_points_path"]
num_classes = config["num_classes"]
target_sample_rate = int(config["target_sample_rate"])
target_time = float(config.get("target_time", 3.0))  # модель обучалась на target_time секунд

# Параметры спектрограммы (как в датасете)
n_mels = 80
n_fft = 1024
hop_length = 256

# Маппинг эмоций
emotion_to_label = {
    "neutral": 0,
    "sad": 1,
    "angry": 2,
    "positive": 3
}
label_to_emotion = {v: k for k, v in emotion_to_label.items()}

# =========================================================
# Ограничения (оставил как было; если хочешь убрать — скажи)
# =========================================================
MIN_AUDIO_DURATION = 0.5
MAX_AUDIO_DURATION = 600
MAX_CONCURRENT_TASKS = 3
TASK_CLEANUP_TIME = 3600
MAX_FILE_SIZE = 50 * 1024 * 1024

# Where to store generated artifacts (CSV/XLSX) for download
TASK_FILES_DIR = Path("./task_artifacts")
TASK_FILES_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Model
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 9
model_path = f"checkpoints/base_model/model_check_points/check_point_{epoch_num}.pth"

model = EmoModel(num_classes=num_classes)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print(f"Модель загружена на {device}")
print(f"target_time (train chunk) из конфига: {target_time} сек")
print(f"sample_rate target: {target_sample_rate} Hz")
print(f"hop_length: {hop_length}, n_fft: {n_fft}, n_mels: {n_mels}")

# =========================================================
# Transforms
# =========================================================
mel_transform = T.MelSpectrogram(
    sample_rate=target_sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)


# =========================================================
# Executor
# =========================================================
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)

# =========================================================
# FastAPI app + storage
# =========================================================
tasks_storage: Dict[str, Dict] = {}
task_queue: asyncio.Queue = asyncio.Queue()
active_tasks: int = 0


# =========================================================
# Pydantic models (оставил)
# =========================================================
class Segment(BaseModel):
    label: str
    startFrame: int
    endFrame: int


class FeaturesRow(BaseModel):
    name: str
    values: List[float]


class FeaturesTable(BaseModel):
    columns: List[str]
    rows: List[FeaturesRow]


class SpectrogramData(BaseModel):
    type: str = "mel"
    frames: int
    melBins: int
    hopLength: int
    minDb: int
    maxDb: int
    format: str = "uint8"
    data: str


# =========================================================
# Utils: conversion
# =========================================================
def convert_to_wav(input_path: str) -> str:
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = output_file.name
    output_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(target_sample_rate),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        output_path,
    ]

    r = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: ({r.returncode}):\n{r.stderr}")
    if (not os.path.exists(output_path)) or (os.path.getsize(output_path) < 1024):
        raise RuntimeError("ffmpeg produced empty or invalid wav")

    return output_path


# =========================================================
# Lifespan
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_old_tasks())
    queue_task = asyncio.create_task(process_task_queue())
    print("✓ Фоновые задачи запущены: очистка старых задач и обработка очереди")

    yield

    cleanup_task.cancel()
    queue_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    try:
        await queue_task
    except asyncio.CancelledError:
        pass
    executor.shutdown(wait=True)
    print("✓ Фоновые задачи остановлены")


app = FastAPI(
    title="Emotion Recognition API",
    version="3.2.0",
    description="API для распознавания эмоций в аудио с использованием CNN",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Cleanup old tasks
# =========================================================
async def cleanup_old_tasks():
    while True:
        try:
            await asyncio.sleep(300)
            now = datetime.now()
            to_delete = []
            for tid, td in tasks_storage.items():
                created_at = td.get("created_at")
                if created_at:
                    age = (now - created_at).total_seconds()
                    if age > TASK_CLEANUP_TIME:
                        to_delete.append(tid)

            for tid in to_delete:
                td = tasks_storage.get(tid, {})
                fp = td.get("features_path")
                if fp:
                    try:
                        Path(fp).unlink(missing_ok=True)
                    except Exception:
                        pass
                del tasks_storage[tid]
                print(f"Очищена старая задача: {tid}")

        except Exception as e:
            print(f"Ошибка очистки задач: {e}")


# =========================================================
# Queue processor
# =========================================================
async def process_task_queue():
    global active_tasks
    while True:
        try:
            task_data = await task_queue.get()

            while active_tasks >= MAX_CONCURRENT_TASKS:
                await asyncio.sleep(0.2)

            active_tasks += 1
            asyncio.create_task(
                process_audio_with_semaphore(
                    task_data["task_id"],
                    task_data["waveform"],
                    task_data["sample_rate"],
                    task_data["window_ms"],
                    task_data["stride_ms"],
                    task_data.get("model_name"),
                )
            )
            task_queue.task_done()

        except Exception as e:
            print(f"Ошибка в обработчике очереди: {e}")


async def process_audio_with_semaphore(
    task_id: str,
    waveform: torch.Tensor,
    sample_rate: int,
    window_ms: int,
    stride_ms: int,
    model_name: Optional[str],
):
    global active_tasks
    try:
        await process_audio_tensor(task_id, waveform, sample_rate, window_ms, stride_ms, model_name)
    finally:
        active_tasks -= 1


# =========================================================
# Helpers: base64 + features formatting
# =========================================================
def numpy_to_base64(array: np.ndarray) -> str:
    flat_array = array.flatten()
    array_bytes = flat_array.tobytes()
    return base64.b64encode(array_bytes).decode("utf-8")


def format_opensmile_features(waveform: torch.Tensor, sample_rate: int, features_dict: Dict[str, float] = None) -> str:
    if features_dict is None:
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze().cpu().numpy()
        else:
            waveform_np = waveform

        df_features = smile.process_signal(waveform_np, sample_rate)
    else:
        import pandas as pd
        df_features = pd.DataFrame([features_dict])

    return df_features.to_csv(index=False)


def write_opensmile_features_csv(task_id: str, waveform: torch.Tensor, sample_rate: int) -> Path:
    """Compute OpenSMILE features and write them to a CSV file on disk.
    Returns the file path.
    """
    csv_text = format_opensmile_features(waveform, sample_rate)
    out_path = TASK_FILES_DIR / f"{task_id}_features.csv"
    out_path.write_text(csv_text, encoding="utf-8")
    return out_path


# =========================================================
# Dataset-compatible preprocessing (ВАЖНО)
# =========================================================
def _crop_or_pad_waveform(waveform: torch.Tensor, target_len: int, mode: str = "center") -> torch.Tensor:
    """
    Привести waveform к длине target_len сэмплов.
    mode:
      - "random"  как в AudioDataset (train)
      - "center"  детерминированно (лучше для инференса)
    """
    cur_len = int(waveform.shape[1])

    if cur_len > target_len:
        if mode == "random":
            start_max = cur_len - target_len
            start = random.randint(0, start_max)
        else:
            start = (cur_len - target_len) // 2
        return waveform[:, start:start + target_len]

    if cur_len < target_len:
        pad = target_len - cur_len
        return torch.nn.functional.pad(waveform, (0, pad))

    return waveform


def _mel_to_model_input(mel_spec: torch.Tensor) -> torch.Tensor:
    """
    mel_spec: [1, 80, T]
    как в AudioDataset:
      log(mel+1e-6)
      z-score
      -> [1, 1, 80, T]
    """
    mel = torch.log(mel_spec + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    elif mel.dim() == 2:
        mel = mel.unsqueeze(0).unsqueeze(0)

    return mel


def _sample_to_frame(sample_idx: int) -> int:
    # грубая привязка: один mel-frame примерно каждые hop_length сэмплов
    return int(sample_idx // hop_length)


# =========================================================
# API: upload
# =========================================================
@app.post("/api/upload")
async def upload_audio(
    audio: UploadFile = File(..., description="Аудио файл (wav, mp3, flac, ogg, m4a, webm, opus)"),
    test_mode: Optional[bool] = Form(False),

    # ВАЖНО: параметры окна приходят с фронта
    window_ms: Optional[int] = Form(None),
    stride_ms: Optional[int] = Form(None),

    # если фронт шлет modelName/model — примем, но сейчас не используем
    model_name: Optional[str] = Form(None),
):
    """
    Загружаешь аудио. Бэк:
      - ресемплит в target_sample_rate
      - считает spectrogram по всей записи
      - гоняет модель по слайдинговым окнам (window_ms/stride_ms)
      - каждое окно приводится к train chunk (target_time) и идет в модель
    """
    try:
        content = await audio.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Файл слишком большой. Максимум: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
            )

        task_id = str(uuid.uuid4())

        # дефолты, если фронт не прислал
        if window_ms is None:
            window_ms = int(target_time * 1000)
        if stride_ms is None:
            stride_ms = int(window_ms)  # по умолчанию без overlap

        # никаких “защит”/эвристик — берем как есть
        window_ms = int(window_ms)
        stride_ms = int(stride_ms)

        if test_mode:
            tasks_storage[task_id] = create_mock_task_result(task_id)
            tasks_storage[task_id]["window_ms"] = window_ms
            tasks_storage[task_id]["stride_ms"] = stride_ms
            tasks_storage[task_id]["model"] = model_name
            return {"ok": True, "taskId": task_id, "status": "done", "message": "Test mode: mock data returned"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            file_suffix = Path(audio.filename).suffix.lower()

            if file_suffix in [".webm", ".ogg", ".opus"]:
                converted_path = convert_to_wav(tmp_path)
                waveform, sample_rate = torchaudio.load(converted_path)
                os.unlink(converted_path)
            else:
                waveform, sample_rate = torchaudio.load(tmp_path)

            duration = float(waveform.shape[1] / sample_rate)

            if duration < MIN_AUDIO_DURATION:
                raise HTTPException(status_code=400, detail=f"Аудио слишком короткое: {duration:.2f} сек")
            if duration > MAX_AUDIO_DURATION:
                raise HTTPException(status_code=400, detail=f"Аудио слишком длинное: {duration/60:.2f} мин")

            # mono
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            print(f"\n{'=' * 60}")
            print(f"Новая задача: {task_id}")
            print(f"Файл: {audio.filename}")
            print(f"Размер: {len(content) / 1024:.1f} KB")
            print(f"Waveform: {tuple(waveform.shape)}")
            print(f"Sample rate: {sample_rate} Hz")
            print(f"Duration: {duration:.2f} sec")
            print(f"window_ms={window_ms}, stride_ms={stride_ms}, model={model_name}")
            print(f"Очередь: {task_queue.qsize()} | Активных: {active_tasks}/{MAX_CONCURRENT_TASKS}")
            print(f"{'=' * 60}")

        finally:
            os.unlink(tmp_path)

        tasks_storage[task_id] = {
            "taskId": task_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.now(),
            "filename": audio.filename,
            "duration": duration,
            "window_ms": window_ms,
            "stride_ms": stride_ms,
            "model": model_name,
        }

        await task_queue.put({
            "task_id": task_id,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "window_ms": window_ms,
            "stride_ms": stride_ms,
            "model_name": model_name,
        })

        return {
            "ok": True,
            "taskId": task_id,
            "status": "queued",
            "queuePosition": task_queue.qsize(),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks_storage[task_id]
    return {k: v for k, v in task.items() if k not in ["waveform", "sample_rate", "created_at", "features_path"]}

@app.get("/api/task/{task_id}/features.csv")
async def download_features_csv(task_id: str):
    """Download OpenSMILE features CSV for a finished task."""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]
    fp = task.get("features_path")
    if not fp:
        raise HTTPException(status_code=404, detail="Features file not ready")

    p = Path(fp)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Features file missing")

    return FileResponse(
        path=str(p),
        media_type="text/csv; charset=utf-8",
        filename=p.name,
    )


# =========================================================
# Main processing
# =========================================================
async def process_audio_tensor(
    task_id: str,
    waveform: torch.Tensor,
    sample_rate: int,
    window_ms: int,
    stride_ms: int,
    model_name: Optional[str],
):
    try:
        tasks_storage[task_id]["status"] = "processing"
        tasks_storage[task_id]["progress"] = 0.05

        loop = asyncio.get_event_loop()

        # mono (на всякий)
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 1) resample exactly as AudioDataset does: torchaudio.functional.resample
        if sample_rate != target_sample_rate:
            waveform = await loop.run_in_executor(executor, F.resample, waveform, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        tasks_storage[task_id]["progress"] = 0.10

        # 2) compute FULL spectrogram once (for frontend + frame timeline)
        # IMPORTANT: this is NOT z-score; frontend spectrogram should look stable across whole file
        full_mel = await loop.run_in_executor(executor, mel_transform, waveform)  # [1, 80, T_full]
        full_mel_log = torch.log(full_mel + 1e-6)  # [1, 80, T_full]
        total_frames_full = int(full_mel_log.shape[2])

        tasks_storage[task_id]["progress"] = 0.18

        # 3) sliding windows over waveform in SAMPLES
        win_len = int(sample_rate * (window_ms / 1000.0))
        hop_len = int(sample_rate * (stride_ms / 1000.0))

        if win_len <= 0:
            win_len = int(sample_rate * target_time)
        if hop_len <= 0:
            hop_len = win_len

        total_len = int(waveform.shape[1])
        if total_len <= 0:
            raise RuntimeError("Empty waveform")

        # train chunk size (3 sec)
        train_len = int(sample_rate * target_time)

        # generate start positions to cover whole audio
        starts = list(range(0, max(total_len - 1, 1), hop_len))
        if len(starts) == 0:
            starts = [0]
        # force last window to cover tail
        if starts[-1] + win_len < total_len:
            starts.append(max(total_len - win_len, 0))

        n_windows = len(starts)

        segments: List[Dict] = []
        predictions: List[str] = []

        print(f"\n{'=' * 60}")
        print(f"PROCESS {task_id}")
        print(f"sr={sample_rate}Hz, samples={total_len}, dur={total_len/sample_rate:.2f}s")
        print(f"window_ms={window_ms} -> win_len={win_len} samples")
        print(f"stride_ms={stride_ms} -> hop_len={hop_len} samples")
        print(f"train_len={train_len} samples (target_time={target_time}s)")
        print(f"full_mel_frames={total_frames_full}")
        print(f"windows={n_windows}")
        print(f"{'=' * 60}")

        # 4) run inference per window
        with torch.no_grad():
            for idx, s0 in enumerate(starts):
                s1 = min(s0 + win_len, total_len)

                window_wave = waveform[:, s0:s1]

                # IMPORTANT: model trained on train_len; so window is mapped to train chunk
                # (center-crop/pad). This is the most correct way without retraining.
                window_wave_fixed = _crop_or_pad_waveform(window_wave, train_len, mode="center")

                mel_seg = await loop.run_in_executor(executor, mel_transform, window_wave_fixed)  # [1,80,T_train]
                x = _mel_to_model_input(mel_seg).to(device)

                logits = model(x)  # IMPORTANT: not in executor
                probs = torch.softmax(logits, dim=1)
                pred_class = int(torch.argmax(probs, dim=1).item())
                emotion_label = label_to_emotion[pred_class]

                predictions.append(emotion_label)

                # map window sample range -> global mel frame range
                f0 = _sample_to_frame(int(s0))
                f1 = _sample_to_frame(int(s1))
                if f0 < 0:
                    f0 = 0
                if f1 < 0:
                    f1 = 0
                if f0 > total_frames_full:
                    f0 = total_frames_full
                if f1 > total_frames_full:
                    f1 = total_frames_full
                if f1 <= f0:
                    f1 = min(f0 + 1, total_frames_full)

                segments.append({"label": emotion_label, "startFrame": int(f0), "endFrame": int(f1)})

                # progress (0.18 .. 0.70)
                tasks_storage[task_id]["progress"] = 0.18 + 0.52 * ((idx + 1) / max(n_windows, 1))

        # 5) summary counts
        emotion_counts = {e: 0 for e in emotion_to_label.keys()}
        for p in predictions:
            emotion_counts[p] += 1
        main_emotion = max(emotion_counts, key=emotion_counts.get)

        print("Emotion counts:", emotion_counts, "main:", main_emotion)

        tasks_storage[task_id]["progress"] = 0.75

        # 6) openSMILE on full waveform (ok)
        waveform_np = waveform.squeeze(0).cpu().numpy()
        features_df = await loop.run_in_executor(executor, smile.process_signal, waveform_np, sample_rate)
        features_data_descriptor = {col: float(features_df[col].iloc[0]) for col in features_df.columns}

        tasks_storage[task_id]["progress"] = 0.88

        # 7) spectrogram for frontend: full_mel_log -> uint8
        mel_for_display = full_mel_log.squeeze(0).transpose(0, 1).cpu().numpy()  # [T_full, 80]

        mel_min = float(mel_for_display.min())
        mel_max = float(mel_for_display.max())
        if mel_max > mel_min:
            mel_uint8 = ((mel_for_display - mel_min) / (mel_max - mel_min) * 255.0).astype(np.uint8)
        else:
            mel_uint8 = np.zeros_like(mel_for_display, dtype=np.uint8)

        mel_base64 = numpy_to_base64(mel_uint8)
        frames, mel_bins = mel_uint8.shape

        tasks_storage[task_id]["progress"] = 0.96

        # 8) finalize
        tasks_storage[task_id].update({
            "status": "done",
            "progress": 1.0,
            "summary": {"mainEmotion": main_emotion, "emotionCounts": emotion_counts},
            "segments": segments,
            "featuresFile": {
                "type": "csv",
                "url": f"/api/task/{task_id}/features.csv",
                "filename": f"{task_id}_features.csv",
            },
            "features_path": str(write_opensmile_features_csv(task_id, waveform, sample_rate)),
            "featuresDataDescriptor": features_data_descriptor,
            "spectrogram": {
                "type": "mel",
                "frames": int(frames),
                "melBins": int(mel_bins),
                "hopLength": int(hop_length),
                "minDb": -80,
                "maxDb": 0,
                "format": "uint8",
                "data": mel_base64,
            }
        })

        print(f"DONE {task_id}: segments={len(segments)}, spec={frames}x{mel_bins}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        tasks_storage[task_id]["status"] = "error"
        tasks_storage[task_id]["error"] = str(e)
        tasks_storage[task_id]["progress"] = 0.0


# =========================================================
# Mock data (оставил, чтобы тестить фронт)
# =========================================================
MOCK_SPECTROGRAM_BASE64 = (
    "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BB"
    "QkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn+AgYKD"
    "hIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8TF"
    "xsfIycrLzM3Oz9DR0tPU1dbX2Nna29zd3t/g4eLj5OXm5+jp6uvs7e7v8PHy8/T19vf4+fr7/P3+"
)


def create_mock_task_result(task_id: str) -> Dict:
    mock_segments = [
        {"label": "neutral", "startFrame": 0, "endFrame": 100},
        {"label": "positive", "startFrame": 100, "endFrame": 200},
        {"label": "neutral", "startFrame": 200, "endFrame": 300},
    ]

    mock_features = {
        "F0semitoneFrom27.5Hz_sma3nz_amean": 0.5234,
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": 0.1234,
        "loudness_sma3_amean": 0.7823,
        "spectralFlux_sma3_amean": 0.3421,
        "mfcc1_sma3_amean": -12.3456,
    }

    return {
        "taskId": task_id,
        "status": "done",
        "progress": 1.0,
        "created_at": datetime.now(),
        "filename": "mock_audio.wav",
        "duration": 3.0,
        "summary": {
            "mainEmotion": "neutral",
            "emotionCounts": {"neutral": 2, "sad": 0, "angry": 0, "positive": 1},
        },
        "segments": mock_segments,
        "featuresFile": {"type":"csv","url":"/api/task/mock/features.csv","filename":"mock_features.csv"},
        "featuresDataDescriptor": mock_features,
        "spectrogram": {
            "type": "mel",
            "frames": 300,
            "melBins": 80,
            "hopLength": hop_length,
            "minDb": -80,
            "maxDb": 0,
            "format": "uint8",
            "data": MOCK_SPECTROGRAM_BASE64,
        },
    }


# =========================================================
# Service endpoints
# =========================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "tasks_count": len(tasks_storage),
        "tasks_active": active_tasks,
        "tasks_queued": task_queue.qsize(),
        "config": {
            "num_classes": num_classes,
            "sample_rate": target_sample_rate,
            "target_time_train": target_time,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "emotions": list(emotion_to_label.keys()),
            "checkpoint": str(Path(model_path).name),
            "opensmile_feature_set": "eGeMAPSv02",
        },
    }


@app.get("/stats")
async def get_stats():
    return {
        "status": "healthy",
        "tasks_count": len(tasks_storage),
        "tasks_active": active_tasks,
        "tasks_queued": task_queue.qsize(),
        "tasks_completed": len([t for t in tasks_storage.values() if t["status"] == "done"]),
        "tasks_failed": len([t for t in tasks_storage.values() if t["status"] == "error"]),
        "tasks_processing": len([t for t in tasks_storage.values() if t["status"] == "processing"]),
        "tasks_queued_list": len([t for t in tasks_storage.values() if t["status"] == "queued"]),
    }


@app.get("/")
async def root():
    return {
        "message": "Emotion Recognition API",
        "model": "EmoModel (CNN)",
        "version": "3.2.0",
        "emotions": list(emotion_to_label.keys()),
        "endpoints": {
            "upload": "POST /api/upload",
            "task_status": "GET /api/task/{task_id}",
            "delete_task": "DELETE /api/task/{task_id}",
            "list_tasks": "GET /api/tasks",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage.get(task_id, {})
    fp = task.get("features_path")
    if fp:
        try:
            Path(fp).unlink(missing_ok=True)
        except Exception:
            pass

    del tasks_storage[task_id]
    return {"ok": True, "message": f"Task {task_id} deleted"}


@app.get("/api/tasks")
async def list_tasks():
    return {
        "total": len(tasks_storage),
        "active": active_tasks,
        "queued": task_queue.qsize(),
        "tasks": [
            {
                "taskId": task.get("taskId"),
                "status": task.get("status"),
                "progress": task.get("progress", 0),
                "filename": task.get("filename", "unknown"),
                "duration": task.get("duration", 0),
                "window_ms": task.get("window_ms"),
                "stride_ms": task.get("stride_ms"),
                "model": task.get("model"),
            }
            for task in tasks_storage.values()
        ],
    }


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("EMOTION RECOGNITION API v3.2.0 (FULL AUDIO WINDOWS)")
    print("=" * 70)
    print(f"Модель: EmoModel (CNN)")
    print(f"Классов: {num_classes}")
    print(f"Эмоции: {', '.join(emotion_to_label.keys())}")
    print(f"Устройство: {device}")
    print(f"Чекпоинт: {model_path}")
    print(f"Train chunk (target_time): {target_time} сек")
    print(f"Sample rate: {target_sample_rate} Hz")
    print(f"API адрес: http://localhost:8080")
    print(f"Документация: http://localhost:8080/docs")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
