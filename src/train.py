import os
import numpy as np
from glob import glob
import joblib
import librosa

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
PROCESSED_PATH = "data/processed/"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------
# FEATURE EXTRACTION
# -------------------------
def extract_features(file_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    features = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)

    S = np.abs(librosa.stft(y))
    column_sums = np.sum(S, axis=0, keepdims=True)
    column_sums[column_sums == 0] = 1e-6  # Prevent divide-by-zero
    ps = S / column_sums

    entropy = -np.sum(ps * np.log2(ps + 1e-6), axis=0)
    entropy = np.nan_to_num(entropy, nan=0.0)

    flux = librosa.onset.onset_strength(y=y, sr=sr)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.max(pitches, axis=0)
    pitch = np.nan_to_num(pitch, nan=0.0)  # Replace NaN with 0

    features.update({
        'mfcc_mean': np.mean(mfcc, axis=1),
        'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
        'entropy_mean': np.mean(entropy),
        'flux_mean': np.mean(flux),
        'pitch_mean': np.mean(pitch),
    })

    features_vector = np.concatenate([
        features['mfcc_mean'],
        features['mfcc_delta_mean'],
        [features['entropy_mean']],
        [features['flux_mean']],
        [features['pitch_mean']]
    ])

    if np.any(np.isnan(features_vector)):
        print(f"NaN feature vector: {features_vector}")
        raise ValueError(f"NaN detected in features for file: {file_path}")

    return features_vector


# -------------------------
# BINARY CLASSIFIER DATA LOADER
# -------------------------

def load_binary_dataset_balanced_split(test_size=0.2, random_state=42):
    human_dir = os.path.join(PROCESSED_PATH, "human", "voice")
    nonhuman_dir = os.path.join(PROCESSED_PATH, "non_human")

    human_files = glob(os.path.join(human_dir, "*.wav"))
    nonhuman_files = []
    for cls in os.listdir(nonhuman_dir):
        cls_path = os.path.join(nonhuman_dir, cls)
        nonhuman_files.extend(glob(os.path.join(cls_path, "*.wav")))

    # Create combined list and labels
    all_files = human_files + nonhuman_files
    all_labels = [1] * len(human_files) + [0] * len(nonhuman_files)

    # Stratified split by samples (not groups)
    train_files, test_files, y_train, y_test = train_test_split(
        all_files, all_labels, test_size=test_size, stratify=all_labels, random_state=random_state
    )

    # Extract features for train set
    X_train, y_train_final = [], []
    for f, label in zip(train_files, y_train):
        try:
            feats = extract_features(f)
            X_train.append(feats)
            y_train_final.append(label)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # Extract features for test set
    X_test, y_test_final = [], []
    for f, label in zip(test_files, y_test):
        try:
            feats = extract_features(f)
            X_test.append(feats)
            y_test_final.append(label)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return np.array(X_train), np.array(X_test), np.array(y_train_final), np.array(y_test_final)


# -------------------------
# MULTICLASS CLASSIFIER DATA LOADER
# -------------------------
def load_nonhuman_multiclass_dataset():
    X, y = [], []
    nonhuman_root = os.path.join(PROCESSED_PATH, "non_human")
    class_names = sorted(os.listdir(nonhuman_root))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        files = glob(os.path.join(nonhuman_root, cls, "*.wav"))
        for f in files:
            try:
                X.append(extract_features(f))
                y.append(class_to_idx[cls])
            except Exception as e:
                print(f"Skipping file {f}: {e}")

    return np.array(X), np.array(y), class_to_idx


# -------------------------
# TRAIN BINARY MODEL
# -------------------------
def train_binary_model():
    print("Training binary (human vs. non-human) classifier with balanced sample split...")
    X_train, X_test, y_train, y_test = load_binary_dataset_balanced_split()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "svm_human_vs_nonhuman.joblib"))

    print("Saved binary classifier to svm_human_vs_nonhuman.joblib")
    print("\nClassification Report:\n", classification_report(y_test, pipeline.predict(X_test)))




# -------------------------
# TRAIN MULTICLASS NON-HUMAN MODEL
# -------------------------
def train_nonhuman_classifier():
    print("\nTraining multi-class (non-human categories) classifier...")
    X, y, class_to_idx = load_nonhuman_multiclass_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "svm_nonhuman_classifier.joblib"))
    joblib.dump(class_to_idx, os.path.join(MODEL_DIR, "nonhuman_class_mapping.joblib"))
    print("Saved multi-class classifier and class mapping.")
    print("\nClassification Report:\n", classification_report(y_test, pipeline.predict(X_test)))


# -------------------------
# MAIN ENTRY POINT
# -------------------------
if __name__ == "__main__":
    train_binary_model()
    train_nonhuman_classifier()