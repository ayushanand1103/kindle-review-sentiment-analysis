# 📚 Kindle Review Sentiment Analysis using Word2Vec + Random Forest

This project performs **sentiment analysis** on Kindle book reviews to determine whether a review expresses a **positive** or **negative** opinion.  
It combines **Natural Language Processing (NLP)** and **Machine Learning** techniques — particularly **Word2Vec embeddings** and a **Random Forest Classifier** — to classify sentiments based on review text.

---

## 🧠 Project Overview

- **Goal:** Classify Kindle reviews as Positive or Negative based on their text.  
- **Model:** Random Forest Classifier  
- **Word Embedding:** Word2Vec (trained from scratch on review data)  
- **Dataset:** Kindle Book Reviews (CSV format)  
- **Language Tools:** NLTK and Gensim for text preprocessing and embeddings  

---

## 🧩 Key Steps

1. **Data Preprocessing**
   - Tokenization and Lemmatization using *NLTK*
   - Stopword removal
   - Conversion of review ratings into binary sentiment labels  
     *(0 = Negative if rating < 3, 1 = Positive if rating ≥ 3)*

2. **Word Embedding**
   - Built custom Word2Vec embeddings (`vector_size=300`, `window=5`, `min_count=2`, `sg=1`)
   - Represented each review as the **average** of its word vectors

3. **Model Training**
   - Trained a Random Forest Classifier (`n_estimators=200`, `class_weight='balanced'`)
   - Used 80/20 train-test split

4. **Evaluation**
   - Evaluated model performance using Accuracy, Confusion Matrix, and Classification Report

---

## 📊 Results
Accuracy: 0.80125 (≈ 80.1%)

Confusion Matrix:
[[ 455 351]
[ 126 1468]]

Classification Report:
precision recall f1-score support
0 0.78 0.56 0.66 806
1 0.81 0.92 0.86 1594

Overall Accuracy: 80%
