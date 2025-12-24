import librosa
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("audio_model/audio_model.pkl", "rb"))

# ----------------------------------------------
# Extract MFCC features for prediction
# ----------------------------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except:
        return None

# ----------------------------------------------
# Predict depression from audio
# ----------------------------------------------
def predict_audio(file_path):
    features = extract_features(file_path)
    if features is None:
        return {"error": "Invalid audio file or unsupported format"}

    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # depressed class prob

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
