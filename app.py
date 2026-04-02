from flask import Flask, request, jsonify, render_template
import nltk
import pickle
import os
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import re

app = Flask(__name__)

# Download required NLTK data
def download_nltk_data():
    resources = ['movie_reviews', 'stopwords', 'punkt', 'punkt_tab']
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except:
            pass

download_nltk_data()

# Global model variables
nb_model = None
lr_model = None
vectorizer = None
nb_accuracy = 0
lr_accuracy = 0

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_models.pkl')

def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def train_models():
    """Train Naive Bayes and Logistic Regression on movie reviews."""
    global nb_model, lr_model, vectorizer, nb_accuracy, lr_accuracy

    print("Loading dataset...")
    documents = []
    labels = []

    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            words = ' '.join(movie_reviews.words(fileid))
            documents.append(preprocess_text(words))
            labels.append(1 if category == 'pos' else 0)

    print(f"Dataset loaded: {len(documents)} reviews")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        documents, labels, test_size=0.2, random_state=42
    )

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    nb_accuracy = round(accuracy_score(y_test, nb_preds) * 100, 2)

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    lr_accuracy = round(accuracy_score(y_test, lr_preds) * 100, 2)

    print(f"Naive Bayes Accuracy: {nb_accuracy}%")
    print(f"Logistic Regression Accuracy: {lr_accuracy}%")
    print("Models trained successfully!")

    # Save to disk for faster startup next time
    print("Saving models to disk...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'nb_model': nb_model,
            'lr_model': lr_model,
            'vectorizer': vectorizer,
            'nb_accuracy': nb_accuracy,
            'lr_accuracy': lr_accuracy
        }, f)

def load_models():
    """Load models from disk if available, otherwise train them."""
    global nb_model, lr_model, vectorizer, nb_accuracy, lr_accuracy
    
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained models from disk...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                nb_model = data['nb_model']
                lr_model = data['lr_model']
                vectorizer = data['vectorizer']
                nb_accuracy = data['nb_accuracy']
                lr_accuracy = data['lr_accuracy']
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}. Retraining...")
            train_models()
    else:
        print("No pre-trained models found.")
        train_models()

def analyze_sentiment(text):
    """Analyze sentiment of given text using both models."""
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])

    # Naive Bayes prediction
    nb_pred = nb_model.predict(vec)[0]
    nb_proba = nb_model.predict_proba(vec)[0]
    nb_confidence = round(max(nb_proba) * 100, 1)

    # Logistic Regression prediction
    lr_pred = lr_model.predict(vec)[0]
    lr_proba = lr_model.predict_proba(vec)[0]
    lr_confidence = round(max(lr_proba) * 100, 1)

    # Key words (simple approach: find words in vocab that lean positive/negative)
    words = processed.split()
    vocab = vectorizer.vocabulary_
    present_words = [w for w in words if w in vocab][:8]

    return {
        "naive_bayes": {
            "label": "Positive" if nb_pred == 1 else "Negative",
            "confidence": nb_confidence,
            "accuracy": nb_accuracy
        },
        "logistic_regression": {
            "label": "Positive" if lr_pred == 1 else "Negative",
            "confidence": lr_confidence,
            "accuracy": lr_accuracy
        },
        "keywords": present_words,
        "processed_words": len(words)
    }

@app.route('/')
def index():
    return render_template('index.html', nb_accuracy=nb_accuracy, lr_accuracy=lr_accuracy)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    result = analyze_sentiment(text)
    return jsonify(result)

if __name__ == '__main__':
    print("Initializing Application...")
    load_models()
    print("Starting server...")
    app.run(debug=True, port=5000)
