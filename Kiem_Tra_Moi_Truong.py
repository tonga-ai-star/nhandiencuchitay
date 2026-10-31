"""Kiem tra moi truong Windows Python 3.12 truoc khi thu thap du lieu.

Chay: python Kiem_Tra_Moi_Truong.py
"""
import importlib
import platform
import sys


def main():
    print("=== KIEM TRA MOI TRUONG NHAN DIEN CU CHI TAY ===")
    print("Python:", sys.version.split()[0])
    print("He dieu hanh:", platform.platform())
    print("Trinh thong dich:", sys.executable)
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Project nay duoc chuan bi cho Python 3.12.x, vui long kiem tra Interpreter.")
    if sys.maxsize <= 2**32:
        errors.append("Can Python 64-bit de dung bo thu vien Windows x64.")

    packages = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
        ("pygame", "Pygame"),
    ]
    loaded = {}
    for module_name, description in packages:
        try:
            module = importlib.import_module(module_name)
            loaded[module_name] = module
            print(f"[OK] {description}: {getattr(module, '__version__', 'da cai')}")
        except Exception as exc:
            print(f"[LOI] {description}: {exc}")
            errors.append(f"Khong import duoc {description}: {exc}")

    mp = loaded.get("mediapipe")
    if mp is not None:
        available = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")
        print("MediaPipe Holistic legacy:", "OK" if available else "KHONG CO")
        if not available:
            errors.append("Khong thay mp.solutions.holistic; can mediapipe==0.10.21.")

    tf = loaded.get("tensorflow")
    if tf is not None:
        try:
            from tensorflow.keras import layers
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(30, 1662)),
                layers.LSTM(4),
                layers.Dense(5, activation="softmax"),
            ])
            print("Keras LSTM build:", "OK", model.output_shape)
        except Exception as exc:
            errors.append(f"Loi Keras LSTM: {exc}")

    if errors:
        print("\nCHUA DAT. Loi can sua:")
        for error in errors:
            print(" -", error)
        print("\nThu chay: python -m pip install -r requirements.txt")
        raise SystemExit(1)
    print("\nTHANH CONG! Co the chay Thu_Thap_Data.py (can webcam).")


if __name__ == "__main__":
    main()
