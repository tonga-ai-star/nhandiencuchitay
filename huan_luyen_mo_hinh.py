"""Doc du lieu .npy, chia tap, train LSTM, luu model va bieu do."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from config import ACTIONS, DATA_DIR, FEATURE_COUNT, MODEL_INFO_PATH, MODEL_PATH, RESULTS_DIR, SEQUENCE_LENGTH


def load_dataset():
    samples, labels, people = [], [], []
    for class_id, action in enumerate(ACTIONS):
        folder = DATA_DIR / action
        if not folder.exists():
            print(f"Chua co du lieu: {folder}")
            continue
        for sequence_folder in sorted(p for p in folder.iterdir() if p.is_dir()):
            frame_files = [sequence_folder / f"{i}.npy" for i in range(SEQUENCE_LENGTH)]
            if not all(f.is_file() for f in frame_files):
                print(f"Bo qua chuoi thieu frame: {sequence_folder}")
                continue
            frames = [np.load(p, allow_pickle=False) for p in frame_files]
            array = np.asarray(frames, dtype=np.float32)
            if array.shape != (SEQUENCE_LENGTH, FEATURE_COUNT) or not np.isfinite(array).all():
                print(f"Bo qua chuoi sai du lieu: {sequence_folder}")
                continue
            samples.append(array)
            labels.append(class_id)
            # Do Thu_Thap_Data.py tao ten folder: person__session__index
            people.append(sequence_folder.name.split("__")[0])

    if not samples:
        raise RuntimeError("Chua co du lieu. Hay chay Thu_Thap_Data.py truoc.")
    X = np.stack(samples)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(people)
    print("Du lieu:", X.shape, "Nhan:", dict(Counter(y)))
    missing = [ACTIONS[i] for i in range(len(ACTIONS)) if i not in set(y)]
    if missing:
        raise RuntimeError(f"Thieu du lieu cac nhan: {missing}")
    if min(Counter(y).values()) < 10:
        raise RuntimeError("Moi nhan nen co it nhat 10 chuoi (khuyen nghi 40-60 chuoi).")
    return X, y, groups


def split_dataset(X, y, people):
    # Uu tien tach theo NGUOI THUC HIEN de tranh ro ri du lieu.
    if len(np.unique(people)) >= 4:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, hold_idx = next(outer.split(X, y, groups=people))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
        val_pos, test_pos = next(inner.split(X[hold_idx], y[hold_idx], groups=people[hold_idx]))
        val_idx, test_idx = hold_idx[val_pos], hold_idx[test_pos]
        mode = "theo_nguoi_thuc_hien"
        print("Chia theo nguoi: train", set(people[train_idx]),
              "val", set(people[val_idx]), "test", set(people[test_idx]))
    else:
        print("CANH BAO: duoi 4 nguoi, tam chia theo chuoi."
              " Ket qua test co the lac quan hon thuc te!")
        all_idx = np.arange(len(y))
        train_idx, hold_idx = train_test_split(
            all_idx, test_size=0.30, random_state=42, stratify=y)
        val_idx, test_idx = train_test_split(
            hold_idx, test_size=0.50, random_state=43, stratify=y[hold_idx])
        mode = "theo_chuoi"
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx], mode)


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_COUNT)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history):
    for key, other, filename, title in [
        ("accuracy", "val_accuracy", "accuracy.png", "Do chinh xac"),
        ("loss", "val_loss", "loss.png", "Sai so (Loss)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history[key], label="Train")
        ax.plot(history.history[other], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / filename, dpi=160)
        plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(42)
    X, y, people = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, split_mode = split_dataset(X, y, people)
    print("Train/Validation/Test:", len(X_train), len(X_val), len(X_test))
    model = build_model(len(ACTIONS))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=16,
        callbacks=callbacks, verbose=1,
    )
    save_history(history)

    best = tf.keras.models.load_model(MODEL_PATH)
    scores = best.predict(X_test, verbose=0)
    predictions = np.argmax(scores, axis=1)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test, predictions,
        labels=np.arange(len(ACTIONS)), target_names=ACTIONS, zero_division=0)
    print(f"\nTEST ACCURACY = {accuracy:.4f}")
    print(report)
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=np.arange(len(ACTIONS)),
        display_labels=ACTIONS, xticks_rotation=45, ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    info = {
        "actions": ACTIONS,
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURE_COUNT,
        "split_mode": split_mode,
        "train_count": len(X_train),
        "val_count": len(X_val),
        "test_count": len(X_test),
        "test_accuracy": accuracy,
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DA LUU MODEL:", MODEL_PATH)
    print("DA LUU BIEU DO:", RESULTS_DIR)


if __name__ == "__main__":
    main()
