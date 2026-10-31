"""Trich xuat landmark MediaPipe va hien thi chu tieng Viet."""
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def detect_landmarks(frame, holistic):
    """Nhan frame BGR, tra ve ket qua MediaPipe Holistic."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = holistic.process(rgb)
    return result


def extract_keypoints(results):
    """1662 gia tri: pose 132 + face 1404 + 2 ban tay 126."""
    pose = (np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark],
                     dtype=np.float32).flatten()
            if results.pose_landmarks else np.zeros(132, dtype=np.float32))
    face = (np.array([[p.x, p.y, p.z] for p in results.face_landmarks.landmark],
                     dtype=np.float32).flatten()
            if results.face_landmarks else np.zeros(1404, dtype=np.float32))
    left = (np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark],
                     dtype=np.float32).flatten()
            if results.left_hand_landmarks else np.zeros(63, dtype=np.float32))
    right = (np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
             if results.right_hand_landmarks else np.zeros(63, dtype=np.float32))
    values = np.concatenate([pose, face, left, right])
    if values.shape != (1662,):
        raise ValueError(f"Sai chieu vector: {values.shape}; can (1662,)")
    return values


def draw_landmarks(frame, result):
    """Ve cac diem co the va tay; khong ve face de man hinh de nhin."""
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def open_camera(index=0):
    """Uu tien DirectShow tren Windows; fallback camera mac dinh."""
    import os
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc webcam. Kiem tra camera, quyen truy cap va CAMERA_INDEX.")
    return cap


def _font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_MAIN = _font(28)
FONT_SMALL = _font(20)


def write_vietnamese(frame, text, pos=(20, 30), big=True):
    """OpenCV khong ve dau tieng Viet tot; dung Pillow cho dong chu."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(image)
    drawer.text(pos, text, font=FONT_MAIN if big else FONT_SMALL,
                fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
