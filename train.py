# train_model.py

import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import warnings

warnings.filterwarnings('ignore')

# --- Download NLTK data (only need to do this once) ---
print("Downloading NLTK data...")
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
print("NLTK data is available.")

# --- Text Cleaning Function ---
# This function combines the cleaning steps from the notebook.
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# --- Training ---
print("Starting model training process...")

# 1. Load the training data
try:
    train_df = pd.read_csv('train.csv')
    print("train.csv loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure it's in the same directory.")
    exit()

# 2. Clean and preprocess the synopsis
print("Cleaning and preprocessing text data...")
train_df['plot'] = train_df['synopsis'].apply(clean_text)
print("Text preprocessing complete.")

# 3. Vectorize the text data using TF-IDF
print("Vectorizing text data with TF-IDF...")
# Using max_features to keep the vocabulary size manageable, which is a good practice.
tfidf_vectorizer = TfidfVectorizer(max_features=10000) 
X_train = tfidf_vectorizer.fit_transform(train_df['plot'])
y_train = train_df['genre']
print("Vectorization complete.")

# 4. Train the classifier (using the best alpha from your notebook)
print("Training the Multinomial Naive Bayes classifier...")
# The notebook's GridSearchCV found alpha=2.0 to be optimal.
classifier = MultinomialNB(alpha=2.0)
classifier.fit(X_train, y_train)
print("Classifier training complete.")

# 5. Save the trained vectorizer and classifier to disk
print("Saving artifacts...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("Training finished. Artifacts 'vectorizer.pkl' and 'model.pkl' are saved.")