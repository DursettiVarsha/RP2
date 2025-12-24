Save the folder as depression_detection
depression_detection/
│
├── data/
│   ├── text/
│   └── audio/
│
├── text_model/
│   ├── train_text.py
│   └── predict_text.py
│
├── audio_model/
│   ├── train_audio.py
│   └── predict_audio.py
│
├── fusion/
│   └── train_fusion_model.py
│
├── app.py   # final UI / CLI
└── requirements.txt

open terminal,
run python text_model/train_text_model.py
It:
• Loads text dataset
• Cleans + TF-IDF vectorizes
• Trains Logistic Regression
• Saves:
──text_model/text_model.pkl
──text_model/tfidf.pkl
──text_model/X_text_test.pkl
──text_model/y_text_test.pkl

then, run
python audio_model/train_audio_model.py
It:
• Extracts MFCC features
• Trains SVM
• Saves:
──audio_model/audio_model.pkl
──audio_model/X_audio_test.pkl
──audio_model/y_audio_test.pkl

Now run the project, in terminal
└──python app.py
In terminal, run:
python fusion/train_fusion_model.py
to know the prediction accuracy of the model
