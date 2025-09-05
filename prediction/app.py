from flask import Flask, request, jsonify
import joblib, re, string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os


svm_model = joblib.load("models/ph_long_svm_model.pkl")
vectorizer = joblib.load("models/final_ph_long_vectorizer.pkl")


# Setup NLTK
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Flask app
app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)

def predict_svm(news_text):
    news_text = clean_text(news_text)
    news_vector = vectorizer.transform([news_text]).toarray()
    prediction = svm_model.predict(news_vector)[0]
    return prediction


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    prediction = predict_svm(text)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)