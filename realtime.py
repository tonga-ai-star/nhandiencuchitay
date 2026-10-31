"""Du doan realtime tu webcam; tuy chon phat MP3 neu co file."""
import json
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from config import (
    CAMERA_INDEX, CONFIDENCE_THRESHOLD, DISPLAY_NAMES,
    MODEL_INFO_PATH, MODEL_PATH, SEQUENCE_LENGTH, SOUND_DIR, STABLE_SECONDS,
)
from utils import detect_landmarks, draw_landmarks, extract_keypoints, open_camera, write_vietnamese


def run(with_sound=False):
    if not MODEL_PATH.is_file() or not MODEL_INFO_PATH.is_file():
        raise RuntimeError("Chua co model. Chay huan_luyen_mo_hinh.py truoc.")

    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    labels = info["actions"]
    if info["features"] != 1662 or info["sequence_length"] != SEQUENCE_LENGTH:
        raise RuntimeError("Thong so model khong khop voi phan nhan dien.")
    model = tf.keras.models.load_model(MODEL_PATH)

    mixer = None
    if with_sound:
        try:
            import pygame
            pygame.mixer.init()
            mixer = pygame.mixer
            print("Phat am thanh qua sounds/<ten_nhan>.mp3")
        except Exception as exc:
            print("Khong khoi tao duoc am thanh:", exc)

    def play_sound(action):
        if mixer is None:
            return
        path = SOUND_DIR / f"{action}.mp3"
        if not path.is_file():
            print("Khong tim thay am thanh:", path.name)
            return
        try:
            mixer.music.load(str(path))
            mixer.music.play()
        except Exception as exc:
            print("Khong phat duoc MP3:", exc)

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    candidate = None
    candidate_since = 0.0
    confirmed = None
    spoken = None
    confidence = 0.0
    frame_count = 0
    cap = open_camera(CAMERA_INDEX)
    try:
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Khong doc duoc webcam")
                    break
                frame = cv2.flip(frame, 1)
                landmarks = detect_landmarks(frame, holistic)
                sequence.append(extract_keypoints(landmarks))
                draw_landmarks(frame, landmarks)
                frame_count += 1

                if len(sequence) == SEQUENCE_LENGTH and frame_count % 3 == 0:
                    sample = np.asarray(sequence, dtype=np.float32)[None, :, :]
                    probabilities = model.predict(sample, verbose=0)[0]
                    index = int(np.argmax(probabilities))
                    confidence = float(probabilities[index])
                    proposed = labels[index] if confidence >= CONFIDENCE_THRESHOLD else None
                    if proposed == "khong_cu_chi":
                        proposed = None

                    now = time.monotonic()
                    if proposed != candidate:
                        candidate = proposed
                        candidate_since = now
                        confirmed = None
                        if proposed is None:
                            spoken = None  # Quay ve trang thai trung tinh => co the phat lai
                    if candidate is not None and now - candidate_since >= STABLE_SECONDS:
                        confirmed = candidate
                        if confirmed != spoken:
                            print(f"Nhan dien: {DISPLAY_NAMES.get(confirmed, confirmed)} ({confidence:.1%})")
                            if with_sound:
                                play_sound(confirmed)
                            spoken = confirmed

                if confirmed:
                    label = DISPLAY_NAMES.get(confirmed, confirmed)
                    frame = write_vietnamese(frame, f"Nhận diện: {label}", (20, 25), big=True)
                else:
                    frame = write_vietnamese(frame, "Đưa tay vào khung hình...", (20, 25), big=True)
                cv2.putText(frame, f"Confidence: {confidence:.2f} | Q: exit", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 250, 60), 2)
                cv2.imshow("NHAN DIEN CU CHI TAY", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mixer:
            mixer.quit()


if __name__ == "__main__":
    run(with_sound=False)
