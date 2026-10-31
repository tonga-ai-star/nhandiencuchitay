"""Cau hinh chung cua project nhan dien cu chi tay."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "MP_Data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SOUND_DIR = ROOT_DIR / "sounds"
MODEL_PATH = MODEL_DIR / "gesture_model.keras"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

ACTIONS = ["khong_cu_chi", "xin_chao", "cam_on", "xin_loi", "tam_biet"]
DISPLAY_NAMES = {
    "khong_cu_chi": "Không có cử chỉ",
    "xin_chao": "Xin chào",
    "cam_on": "Cảm ơn",
    "xin_loi": "Xin lỗi",
    "tam_biet": "Tạm biệt",
}

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 1662
SEQUENCES_PER_ACTION = 5  # Co the tang len 60 sau khi demo duoc
PREPARE_SECONDS = 2.0
CONFIDENCE_THRESHOLD = 0.80
STABLE_SECONDS = 1.0
CAMERA_INDEX = 0

for folder in (DATA_DIR, MODEL_DIR, RESULTS_DIR, SOUND_DIR):
    folder.mkdir(parents=True, exist_ok=True)
