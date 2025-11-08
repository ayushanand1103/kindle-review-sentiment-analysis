# Importing libraries
import pandas as pd 
import numpy as np 
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
import gensim 
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec , KeyedVectors
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Storing refferences 
lemmatizer = WordNetLemmatizer()
tokenizer = sent_tokenize
word_toke = word_tokenize
stop_words = set(stopwords.words("english"))


# Reading dataset
df = pd.read_csv("C:/Users\Ayush\.spyder-py3/ML codes/Kindle book review sentiment analysis using word2vec from scratch/all_kindle_review .csv")

# Data preprocessing and cleaning
Y = df["rating"].apply(lambda x: 0 if x < 3 else 1) # Convert ratings to sentiment: 0 = negative (<3), 1 = positive (>=3)
X = df["reviewText"]
corpus = []
for review in X:
    tokens = simple_preprocess(str(review))
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    corpus.append(lemmatized)

# Train test spilt 
X_train, X_test, Y_train, Y_test = train_test_split(corpus, Y, test_size=0.20)

#Creating Word2Vec model 
model = gensim.models.Word2Vec(X_train, vector_size=300, window=5, min_count=2, sg=1)
#Avgword2vec
def avg_word2vec(doc, model):
    # Get vectors for known words only
    return  np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
X_train_vecs = [avg_word2vec(doc, model) for doc in tqdm(X_train)]
X_new = np.array(X_train_vecs)


    
    
# Convert test data into average Word2Vec vectors
X_test_vecs = [avg_word2vec(doc, model) for doc in tqdm(X_test, desc="Vectorizing test set")]
X_test_vecs = np.array(X_test_vecs)



# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train_vecs, Y_train)


# Predict on test set
y_pred = clf.predict(X_test_vecs)

# Evaluate
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

