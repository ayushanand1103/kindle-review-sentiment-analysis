# kindle-review-sentiment-analysis
# Kindle Review Sentiment Analysis

This project performs sentiment analysis on Kindle book reviews using **Word2Vec** embeddings and a **Random Forest Classifier**.  
The goal is to classify reviews as **positive** or **negative** based on their text content.

## 🔍 Overview
- Preprocessed text using NLTK (lemmatization + stopword removal)
- Generated embeddings using Gensim Word2Vec
- Trained a Random Forest model for classification
- Evaluated using accuracy, confusion matrix, and classification report

## 🧠 Libraries Used
- pandas, numpy
- nltk
- gensim
- scikit-learn
- tqdm

## 📊 Results
The model achieved an accuracy of around **(your accuracy here)** on the test data.

## 🚀 How to Run
1. Install the dependencies:
   ```bash
   pip install pandas numpy nltk gensim scikit-learn tqdm
