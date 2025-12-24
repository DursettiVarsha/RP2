import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


AUDIO_DIR = "data/audio/"

# ----------------------------------------------
# Extract MFCC features
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
# Emotion label mapping
# ----------------------------------------------
def get_label(filename):
    emotion = int(filename.split("-")[2])  # third number
    depressed_emotions = [4, 5, 6, 7]  # sad, angry, fear, disgust

    return 1 if emotion in depressed_emotions else 0


features = []
labels = []

# ----------------------------------------------
# Loop through all audio folders
# ----------------------------------------------
for actor_folder in os.listdir(AUDIO_DIR):
    folder_path = os.path.join(AUDIO_DIR, actor_folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(get_label(file))

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# ----------------------------------------------
# Train-test split
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------
# Train SVM classifier
# ----------------------------------------------
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("audio_model/audio_model.pkl", "wb"))

# Save features for fusion
pickle.dump(X_test, open("audio_model/X_audio_test.pkl", "wb"))
pickle.dump(y_test, open("audio_model/y_audio_test.pkl", "wb"))

print("Audio model saved!")


print("✅ Audio model trained and saved successfully!")
