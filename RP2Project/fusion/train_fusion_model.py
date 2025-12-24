import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load text model
text_model = pickle.load(open("text_model/text_model.pkl", "rb"))

# Load audio model
audio_model = pickle.load(open("audio_model/audio_model.pkl", "rb"))

# -----------------------------------------------------------
# Load datasets (you MUST already have feature files)
# -----------------------------------------------------------

# Text features
X_text_test = pickle.load(open("text_model/X_text_test.pkl", "rb"))
y_text_test = pickle.load(open("text_model/y_text_test.pkl", "rb"))

# Audio features
X_audio_test = pickle.load(open("audio_model/X_audio_test.pkl", "rb"))
y_audio_test = pickle.load(open("audio_model/y_audio_test.pkl", "rb"))

# Make sure labels match
y_test = np.array(y_text_test)

# -----------------------------------------------------------
# Fusion Model
# -----------------------------------------------------------

final_probs = []
final_preds = []

for i in range(len(y_test)):
    text_vector = tfidf.transform([X_text_test[i]])   # Convert string → TF-IDF vector
    text_prob = text_model.predict_proba(text_vector)[0][1]

    audio_prob = audio_model.predict_proba(X_audio_test[i].reshape(1, -1))[0][1]

    fused_prob = (0.6 * text_prob) + (0.4 * audio_prob)
    final_probs.append(fused_prob)

    final_preds.append(1 if fused_prob >= 0.5 else 0)

# -----------------------------------------------------------
# Metrics
# -----------------------------------------------------------

acc = accuracy_score(y_test, final_preds)
prec = precision_score(y_test, final_preds)
rec = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)
auc = roc_auc_score(y_test, final_probs)

print("\n===== FUSION MODEL ACCURACY =====")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)
print("AUC:", auc)

# -----------------------------------------------------------
# Save fusion model as a dictionary
# -----------------------------------------------------------

fusion_model = {
    "text_model": text_model,
    "audio_model": audio_model,
    "text_weight": 0.6,
    "audio_weight": 0.4
}

pickle.dump(fusion_model, open("fusion_model.pkl", "wb"))

print("\nFusion model saved as fusion_model.pkl")
