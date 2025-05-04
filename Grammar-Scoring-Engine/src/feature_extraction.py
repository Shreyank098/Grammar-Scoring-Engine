import numpy as np
import librosa

def extract_features(file_path):
    """Extract MFCC, Chroma, Spectral Contrast, Zero-Crossing Rate, and RMS features from an audio file"""
    audio, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma = np.mean(chroma, axis=1)

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast = np.mean(contrast, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr = np.mean(zcr)

    rms = librosa.feature.rms(y=audio)
    rms = np.mean(rms)

    features = np.hstack([mfcc, chroma, contrast, zcr, rms])
    return features
