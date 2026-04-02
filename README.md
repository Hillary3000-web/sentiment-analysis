# SentimentIQ — CSC 309 Mini Project
**Hillary Chukwuma Prince | Group 2 | Sentiment Analysis System**

---

## What it does
A web-based Sentiment Analysis System that:
- Detects whether text is **Positive** or **Negative**
- Runs **two ML models** side by side (Naïve Bayes + Logistic Regression)
- Shows **confidence scores** for each prediction
- Highlights **keywords** detected in the input
- Displays **training accuracy** of each model
- Indicates whether both models **agree or disagree**
- **Caches trained models** to disk (`.pkl`) for lightning-fast server startup

Trained on the **NLTK Movie Reviews Dataset** (2,000 labeled reviews).

---

## Concepts Covered
- **NLP**: Text preprocessing, tokenization, stopword removal
- **TF-IDF Vectorization**: Converting text to numerical features
- **Naïve Bayes**: Probabilistic classifier (MultinomialNB)
- **Logistic Regression**: Linear classifier with probability output
- **Model Evaluation**: Train/test split, accuracy scoring

---

## Setup & Run

### 1. Install dependencies
```bash
pip install flask nltk scikit-learn numpy
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

> **Note**: On the first run, NLTK will download the movie_reviews corpus (~3MB). The models will take ~30 seconds to train, but are then saved locally. **Subsequent startups will load the pre-trained models instantly.**

---

## Project Structure
```
sentiment_analysis/
├── app.py              # Flask backend + ML logic
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── models/             # Pickled trained ML models (persistent storage)
└── templates/
    └── index.html      # Frontend UI
```

---

## How it works
1. **Preprocessing**: Text is lowercased, punctuation removed, stopwords filtered
2. **Vectorization**: TF-IDF with bigrams converts text to feature vectors
3. **Prediction**: Both models output a label + probability score
4. **Display**: Results shown with confidence bars and keyword highlights

---

*CSC 309 — Artificial Intelligence | FUTO*
