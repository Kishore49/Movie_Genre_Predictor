import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import warnings

warnings.filterwarnings('ignore')

# ⬇️ Tell NLTK to use the local nltk_data directory
nltk.data.path.append('./nltk_data')  # Adjust this if running from outside app folder

# --- Text Cleaning Function ---
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
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train = tfidf_vectorizer.fit_transform(train_df['plot'])
y_train = train_df['genre']
print("Vectorization complete.")

# 4. Train the classifier
print("Training the Multinomial Naive Bayes classifier...")
classifier = MultinomialNB(alpha=2.0)
classifier.fit(X_train, y_train)
print("Classifier training complete.")

# 5. Save the artifacts
print("Saving artifacts...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("Training finished. Artifacts 'vectorizer.pkl' and 'model.pkl' are saved.")
