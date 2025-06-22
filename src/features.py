import librosa
import numpy as np

def extract_features(file_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    features = {}
    
    # MFCC and deltas
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Spectral entropy
    S = np.abs(librosa.stft(y))
    ps = S / np.sum(S, axis=0, keepdims=True)
    entropy = -np.sum(ps * np.log2(ps + 1e-12), axis=0)

    # Spectral flux
    flux = librosa.onset.onset_strength(y=y, sr=sr)

    # Pitch & harmonicity
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.max(pitches, axis=0)
    
    # Feature aggregation (mean, std)
    features.update({
        'mfcc_mean': np.mean(mfcc, axis=1),
        'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
        'entropy_mean': np.mean(entropy),
        'flux_mean': np.mean(flux),
        'pitch_mean': np.mean(pitch),
    })
    return np.concatenate([
        features['mfcc_mean'],
        features['mfcc_delta_mean'],
        [features['entropy_mean']],
        [features['flux_mean']],
        [features['pitch_mean']]
    ])

