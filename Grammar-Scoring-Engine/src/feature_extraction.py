import numpy as np
import librosa

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    rmse = librosa.feature.rms(y=audio)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(zero_crossing),
        np.mean(rmse)
    ])
    return features

