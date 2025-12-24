import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Text Cleaning Function
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
# Load Dataset
# ----------------------------
df = pd.read_csv("data/text/depression_text.csv", low_memory=False)

# Remove rows with missing label
df = df.dropna(subset=["label"])

# Convert label to integer
df["label"] = df["label"].astype(int)

# Combine title + body
df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# ----------------------------
# Split BEFORE TF-IDF
# ----------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

# ----------------------------
# TF-IDF Vectorizer
# ----------------------------
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# ----------------------------
# Model Training
# ----------------------------
model = LogisticRegression(max_iter=3000)
model.fit(X_train_tfidf, y_train)

# ----------------------------
# Save model + TF-IDF
# ----------------------------
pickle.dump(model, open("text_model/text_model.pkl", "wb"))
pickle.dump(tfidf, open("text_model/tfidf.pkl", "wb"))

# ----------------------------
# For fusion model — SAVE RAW TEXT, not vectors
# ----------------------------
pickle.dump(list(X_test_text), open("text_model/X_text_test.pkl", "wb"))
pickle.dump(y_test, open("text_model/y_text_test.pkl", "wb"))

print("Text model saved successfully!")
print("✅ Training Complete! Model Saved.")
