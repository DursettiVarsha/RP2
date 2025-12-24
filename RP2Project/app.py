from flask import Flask, render_template, request
from text_model.predict_text import predict_depression
from audio_model.predict_audio import predict_audio
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["user_text"]
    result = predict_depression(user_text)

    prediction = "Depressed" if result["prediction"] == 1 else "Not Depressed"

    return render_template(
        "result.html",
        user_text=user_text,
        prediction=prediction,
        confidence=round(result["confidence"] * 100, 2)
    )

@app.route("/predict_audio", methods=["POST"])
def predict_audio_route():
    if "audio_file" not in request.files:
        return "No audio file uploaded"

    audio = request.files["audio_file"]

    save_path = "uploaded_audio.wav"
    audio.save(save_path)

    result = predict_audio(save_path)

    if "error" in result:
        return result["error"]

    prediction = "Depressed" if result["prediction"] == 1 else "Not Depressed"

    return render_template(
        "result_audio.html",
        prediction=prediction,
        confidence=round(result["confidence"] * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
