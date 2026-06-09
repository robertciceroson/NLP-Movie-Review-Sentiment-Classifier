# NLP Movie Review Sentiment Classifier

A full NLP pipeline from raw text to transformer-based classification, applied to the IMDB movie review dataset. Covers classical machine learning baselines through modern deep learning — built as part of an applied ML portfolio.

---

## Overview

This project classifies IMDB movie reviews as **positive** or **negative** using a progressive stack of NLP techniques, starting from simple text preprocessing and working up to fine-tuned transformer models. Each approach is benchmarked on the same test set so performance gains are directly comparable.

**Dataset:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — 50,000 labeled reviews (balanced positive / negative). The notebook samples 2,000 reviews (1,000 per class) for a fast, reproducible experiment with an 80/20 stratified train/test split.

---

## Concepts Covered

| Step | Concept |
|---|---|
| Text Preprocessing | Tokenization, stopword removal, stemming (Porter), lemmatization (WordNet) |
| Feature Extraction | Bag-of-Words, TF-IDF with bigrams |
| Classical ML | Logistic Regression, Multinomial Naive Bayes, Linear SVC |
| Word Embeddings | Word2Vec trained from scratch (Skip-gram), GloVe pretrained (6B tokens, 100d) |
| Transformers | DistilBERT fine-tuning with Hugging Face Transformers |
| Evaluation | Accuracy, ROC-AUC, Classification Report, Confusion Matrix, Training Loss curve |

---

## Model Results

All models evaluated on the same 400-sample held-out test set.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression (TF-IDF) | ~0.87 | ~0.94 |
| Naive Bayes (TF-IDF) | ~0.85 | ~0.92 |
| Linear SVC (TF-IDF) | ~0.88 | N/A |
| Logistic Regression (Word2Vec, from scratch) | ~0.79 | ~0.87 |
| Logistic Regression (GloVe 6B 100d, pretrained) | ~0.83 | ~0.90 |
| DistilBERT (fine-tuned) | ~0.91 | ~0.97 |

> Exact values will vary slightly per run due to random seeds and sampling. Run the full notebook to reproduce.

---

## Key Takeaways

- **TF-IDF + LinearSVC** is a strong, fast baseline — competitive with embeddings on small datasets and fully interpretable via feature coefficients
- **Word2Vec trained from scratch** on 1,600 reviews underperforms pretrained GloVe, demonstrating the value of large-corpus pretraining
- **GloVe** (6B tokens) outperforms Word2Vec from scratch by ~3–5% by leveraging broad vocabulary knowledge
- **DistilBERT** achieves the highest accuracy by capturing context-aware representations — handling negation ("not good") and sarcasm that bag-of-words models miss
- Mean-pooling word vectors loses word order; transformers preserve it through attention

---

## Project Structure

```
NLP_Movie_Review_Sentiment_Classifier/
│
├── V3_nlp_sentiment_classifier.ipynb   # Main notebook — full pipeline
├── IMDB_Dataset.csv                    # Dataset (download separately from Kaggle)
├── glove.6B.100d.txt                   # GloVe vectors (auto-downloaded by notebook)
└── README.md
```

---

## Setup & Requirements

### Install dependencies

```bash
pip install transformers torch scikit-learn nltk gensim matplotlib seaborn wordcloud tqdm
```

### Download NLTK data (run once)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### GloVe vectors

The notebook auto-downloads `glove.6B.zip` (~822 MB) from Stanford on first run and extracts `glove.6B.100d.txt`. Ensure you have a stable internet connection and ~1 GB of free disk space when running the GloVe section for the first time.

### DistilBERT / GPU note

The DistilBERT fine-tuning section is computationally intensive. On CPU it will run but may take 20–40 minutes per epoch. Running on Google Colab with GPU runtime is recommended for the BERT section.

---

## Notebook Walkthrough

| Step | Description |
|---|---|
| Step 0 | Install dependencies |
| Step 1 | Imports |
| Step 2 | Load and sample IMDB dataset, stratified train/test split |
| Step 3 | Exploratory Data Analysis — class distribution, review length, word frequency |
| Step 4 | Text preprocessing — HTML removal, tokenization, stopwords, stemming/lemmatization |
| Step 5 | TF-IDF feature extraction with bigrams |
| Step 6 | Classical ML classifiers — Logistic Regression, Naive Bayes, Linear SVC |
| Step 7 | Word embeddings — Word2Vec (from scratch), GloVe (pretrained), t-SNE visualization |
| Step 8 | Transformers — DistilBERT tokenization, fine-tuning, training loss curve |
| Step 9 | Full model comparison — accuracy and ROC-AUC across all approaches |
| Step 10 | Custom text prediction — run any review through all models |

---

## Sample Output — Custom Review Predictions

```
Input: "This film was absolutely stunning. The performances were extraordinary."
  Logistic Regression (TF-IDF) → POSITIVE (0.96)
  Linear SVC                   → POSITIVE
  DistilBERT                   → POSITIVE (0.99)

Input: "Complete waste of time. Terrible plot, awful dialogue."
  Logistic Regression (TF-IDF) → NEGATIVE (0.97)
  Linear SVC                   → NEGATIVE
  DistilBERT                   → NEGATIVE (0.99)
```

---

## Part of ML Portfolio

This project is part of an applied machine learning portfolio covering regression, classification, deep learning, and NLP:

- [Airbnb Price Prediction](https://github.com/robertciceroson/Airbnb_Price_Prediction)
- [Heart Disease Risk Classification](https://github.com/robertciceroson/Heart_Disease_Risk_Prediction)
- [Digit Recognition App (CNN)](https://github.com/robertciceroson/Digit_Recognition_App)
- **NLP Movie Review Sentiment Classifier** ← this repo

---

## Author

**Robert C. Son**  
Process Engineer · Data Analyst · CSM · CSPO · AI-Empowered SAFe Agilist  
[github.com/robertciceroson](https://github.com/robertciceroson) · [LinkedIn](https://www.linkedin.com/in/robert-son-0b33b3bb)
