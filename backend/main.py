from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load trained model
with open("../models/fake_news_model.pkl", "rb") as model_file:
    model_pipeline = pickle.load(model_file)

def clean_text(text):
    """Preprocess input text for prediction."""
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

@app.route("/predict", methods=["POST"])
def predict():
    """Predict whether a news article is fake or real."""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned_text = clean_text(text)
    prediction = model_pipeline.predict([cleaned_text])[0]

    return jsonify({"prediction": "Fake News" if prediction == 1 else "Real News"})

if __name__ == "__main__":
    app.run(debug=True)
