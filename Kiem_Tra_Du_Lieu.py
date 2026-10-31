"""In thong ke so chuoi .npy hop le theo nhan va nguoi tham gia.

Chay: python Kiem_Tra_Du_Lieu.py
"""
from collections import Counter

import numpy as np

from config import ACTIONS, DATA_DIR, FEATURE_COUNT, SEQUENCE_LENGTH


def main():
    total = 0
    print("=== KIEM TRA DU LIEU ===")
    for action in ACTIONS:
        folder = DATA_DIR / action
        count = 0
        bad = 0
        people = Counter()
        if folder.is_dir():
            for sequence in folder.iterdir():
                if not sequence.is_dir():
                    continue
                paths = [sequence / f"{i}.npy" for i in range(SEQUENCE_LENGTH)]
                try:
                    if not all(p.is_file() for p in paths):
                        raise ValueError("Thieu frame")
                    frames = np.asarray([np.load(p, allow_pickle=False) for p in paths], dtype=np.float32)
                    if frames.shape != (SEQUENCE_LENGTH, FEATURE_COUNT) or not np.isfinite(frames).all():
                        raise ValueError("Sai kich thuoc hoac du lieu")
                except (OSError, ValueError, TypeError):
                    bad += 1
                    continue
                count += 1
                people[sequence.name.split("__")[0]] += 1
        print(f"{action:20s} : {count:4d} chuoi hop le, {bad:3d} chuoi loi; nguoi: {dict(people)}")
        total += count
    print("Tong chuoi hop le:", total)
    if total == 0:
        print("Chua co du lieu. Chay Thu_Thap_Data.py truoc.")


if __name__ == "__main__":
    main()
