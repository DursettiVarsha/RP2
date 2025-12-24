import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------------
# Load saved test data
# ----------------------------
X_text_test = pickle.load(open("text_model/X_text_test.pkl", "rb"))
y_text_test = pickle.load(open("text_model/y_text_test.pkl", "rb"))

X_audio_test = pickle.load(open("audio_model/X_audio_test.pkl", "rb"))
y_audio_test = pickle.load(open("audio_model/y_audio_test.pkl", "rb"))

# ----------------------------
# Load trained models
# ----------------------------
text_model = pickle.load(open("text_model/text_model.pkl", "rb"))
tfidf = pickle.load(open("text_model/tfidf.pkl", "rb"))
audio_model = pickle.load(open("audio_model/audio_model.pkl", "rb"))

# ----------------------------
# Generate fusion features
# ----------------------------
fusion_X = []
fusion_y = []

N = min(len(X_text_test), len(X_audio_test))

for i in range(N):
    # TEXT probability
    text_vec = tfidf.transform([X_text_test[i]])
    text_prob = text_model.predict_proba(text_vec)[0][1]

    # AUDIO probability
    audio_prob = audio_model.predict_proba([X_audio_test[i]])[0][1]

    fusion_X.append([text_prob, audio_prob])
    fusion_y.append(y_text_test.iloc[i])


fusion_X = np.array(fusion_X)
fusion_y = np.array(fusion_y)

# ----------------------------
# Train Fusion Classifier
# ----------------------------
fusion_model = LogisticRegression()
fusion_model.fit(fusion_X, fusion_y)

# ----------------------------
# Evaluate Accuracy
# ----------------------------
y_pred = fusion_model.predict(fusion_X)
accuracy = accuracy_score(fusion_y, y_pred)

print("\n🔥 MULTIMODAL FUSION MODEL ACCURACY 🔥")
print("====================================")
print("Accuracy:", round(accuracy * 100, 2), "%")

# ----------------------------
# Save fusion model
# ----------------------------
pickle.dump(fusion_model, open("fusion/fusion_model.pkl", "wb"))

print("\nFusion model saved successfully!")
