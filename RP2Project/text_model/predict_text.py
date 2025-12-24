import pickle
import re
import string
import pandas as pd

# ----------------------------
# Load model + vectorizer
# ----------------------------
tfidf = pickle.load(open("text_model/tfidf.pkl", "rb"))
model = pickle.load(open("text_model/text_model.pkl", "rb"))

# ----------------------------
# Clean incoming text
# ----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------
# Prediction function
# ----------------------------
def predict_depression(user_text):
    cleaned = clean_text(user_text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][1]  # probability of depressed class

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
