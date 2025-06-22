import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from glob import glob
import shutil
import random

ESC50_CSV = "data/ESC_50/meta/esc50.csv"
ESC50_AUDIO = "data/ESC_50/audio/"
TIMIT_PATH = "data/TIMIT/TRAIN/DR2/"
PROCESSED_PATH = "data/processed/"
SELECTED_ESC_CLASSES = [
    "helicopter", "engine", "car_horn", "chainsaw", "crow",
    "door_wood_creaks", "glass_breaking", "pouring_water",
    "rain", "water_drops", "wind"
]

def preprocess_audio(y, target_sr=16000, target_length=16000):
    y, _ = librosa.effects.trim(y, top_db=20)
    y = y / np.max(np.abs(y) + 1e-6)  # normalize
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y

def segment_audio(file_path, label, sublabel, out_dir, sr=16000, frame_dur=1.0):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        print(f"Loaded file: {file_path}, length: {len(y)}")
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    frame_len = int(frame_dur * sr)
    output_dir = os.path.join(out_dir, label, sublabel)
    os.makedirs(output_dir, exist_ok=True)

    speaker_id = os.path.basename(os.path.dirname(file_path))
    utterance_id = os.path.splitext(os.path.basename(file_path))[0]
    base_name = f"audio_1_0_{speaker_id}_{utterance_id}"
    j = 0

    for i in range(0, len(y) - frame_len + 1, frame_len):
        chunk = y[i:i + frame_len]
        chunk = preprocess_audio(chunk, sr, frame_len)
        fname = f"{base_name}_{j}.wav"
        j += 1
        sf.write(os.path.join(output_dir, fname), chunk, sr)



def preprocess_esc50():
    df = pd.read_csv(ESC50_CSV)
    selected = df[df['category'].isin(SELECTED_ESC_CLASSES)]
    for _, row in selected.iterrows():
        print(os.path.join(ESC50_AUDIO, row['filename']))
        segment_audio(
            os.path.join(ESC50_AUDIO, row['filename']),
            "non_human", row['category'].replace(" ", "_"),
            PROCESSED_PATH
        )

def preprocess_timit():
    for root, _, files in os.walk(TIMIT_PATH):
        for f in files:
            if f.endswith(".WAV"):
                print(f"Processing TIMIT file: {f}")
                full_path = os.path.join(root, f)
                print("Processing:", full_path)
                segment_audio(full_path, "human", "voice", PROCESSED_PATH)
            else:
                print(f"Skipping non-wav file: {f}")



def balance_dataset():
    # Get all human samples and shuffle
    human_files = glob(os.path.join(PROCESSED_PATH, "human", "voice", "*.wav"))
    random.shuffle(human_files)

    # Collect non-human files by class
    non_human_samples = []
    for cls in SELECTED_ESC_CLASSES:
        class_path = os.path.join(PROCESSED_PATH, "non_human", cls)
        class_files = glob(os.path.join(class_path, "*.wav"))
        non_human_samples.extend(class_files)

    # Determine target total count for balancing
    n_limit = min(len(human_files), len(non_human_samples))
    print(f"Balancing to {n_limit} samples total for human and non-human combined...")

    # Trim human files to n_limit samples
    for f in human_files[n_limit:]:
        os.remove(f)

    # Calculate limit per non-human subclass
    num_classes = len(SELECTED_ESC_CLASSES)
    per_class_limit = n_limit // num_classes
    print(f"Balancing each non-human class to {per_class_limit} samples...")

    # Trim each non-human subclass
    for cls in SELECTED_ESC_CLASSES:
        class_path = os.path.join(PROCESSED_PATH, "non_human", cls)
        class_files = glob(os.path.join(class_path, "*.wav"))
        random.shuffle(class_files)  # shuffle before trimming
        # Remove files exceeding per_class_limit
        for f in class_files[per_class_limit:]:
            os.remove(f)


def preprocess_all():
    preprocess_esc50()
    preprocess_timit()
    balance_dataset()
