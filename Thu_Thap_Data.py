"""Thu chuoi 30 khung hinh webcam thanh cac tep .npy."""
import re
import shutil
import time

import cv2
import mediapipe as mp
import numpy as np

from config import (
    ACTIONS, CAMERA_INDEX, DATA_DIR, PREPARE_SECONDS,
    SEQUENCE_LENGTH, SEQUENCES_PER_ACTION,
)
from utils import detect_landmarks, draw_landmarks, extract_keypoints, open_camera


class StopCollection(Exception):
    pass


def ask_code(prompt, default):
    raw = input(f"{prompt} (mac dinh {default}): ").strip() or default
    if not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError("Ma chi duoc dung chu khong dau, so, _, -")
    return raw


def next_folder(action, person, session):
    root = DATA_DIR / action
    root.mkdir(parents=True, exist_ok=True)
    idx = 0
    while (root / f"{person}__{session}__{idx:04d}").exists():
        idx += 1
    return root / f"{person}__{session}__{idx:04d}"


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Khong doc duoc frame tu camera")
    return cv2.flip(frame, 1)


def main():
    print("=== THU THAP DU LIEU CU CHI ===")
    person = ask_code("Ma nguoi thuc hien, vi du p01", "p01")
    session = ask_code("Ma buoi quay, vi du s01", "s01")
    print("Nhan Q trong cua so camera de dung. Lam cu chi sau khi het thong bao CHUAN BI.")
    print(f"Moi nhan: {SEQUENCES_PER_ACTION} chuoi x {SEQUENCE_LENGTH} khung hinh")

    cap = open_camera(CAMERA_INDEX)
    window = "THU THAP DU LIEU - Q de thoat"
    try:
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            for action in ACTIONS:
                print(f"\nDang thu: {action}")
                for i in range(SEQUENCES_PER_ACTION):
                    folder = next_folder(action, person, session)
                    start = time.monotonic()
                    while time.monotonic() - start < PREPARE_SECONDS:
                        frame = read_frame(cap)
                        cv2.putText(frame, f"CHUAN BI: {action} ({i + 1}/{SEQUENCES_PER_ACTION})",
                                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
                        cv2.imshow(window, frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise StopCollection()
                    folder.mkdir(parents=True, exist_ok=False)
                    try:
                        for frame_index in range(SEQUENCE_LENGTH):
                            frame = read_frame(cap)
                            results = detect_landmarks(frame, holistic)
                            values = extract_keypoints(results)
                            np.save(folder / f"{frame_index}.npy", values)
                            draw_landmarks(frame, results)
                            cv2.putText(frame, f"DANG QUAY: {action} | Mau {i + 1}/{SEQUENCES_PER_ACTION}",
                                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 50), 2)
                            cv2.putText(frame, f"Frame {frame_index + 1}/{SEQUENCE_LENGTH} - Q de thoat",
                                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 50), 2)
                            cv2.imshow(window, frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                raise StopCollection()
                    except (StopCollection, RuntimeError):
                        shutil.rmtree(folder, ignore_errors=True)  # Xoa chuoi quay do
                        raise
                    print(f"Da luu {action}: {folder.name}")
    except StopCollection:
        print("Da dung theo yeu cau, du lieu hoan chinh truoc do van duoc giu.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
