import streamlit as st
import re
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os

# Add this before you use WordNetLemmatizer or any NLTK corpus
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", download_dir=nltk_data_path)
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", download_dir=nltk_data_path)

st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬", layout="wide")

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

@st.cache_resource
def load_artifacts():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

def clean_text(text, lemmatizer):
    text = re.sub(r'http\S+|[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

st.markdown("""
<style>
    .stTextArea [data-baseweb=base-input] {
        font-size: 1.1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        padding: 10px 0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

try:
    lemmatizer = load_lemmatizer()
    vectorizer, model = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are available in the root directory.")
    st.stop()

st.title("🎬 Movie Genre Predictor")
st.markdown("Predict the genre of a movie based on its plot synopsis.")

user_input = st.text_area(
    "Enter the movie synopsis:",
    height=200,
    placeholder="A young wizard on his eleventh birthday discovers he is, in fact, a wizard and is invited to attend a magical school..."
)

if st.button("Predict Genre", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a synopsis before predicting.")
        st.session_state.prediction_results = None
    elif not hasattr(model, "predict_proba"):
        st.error("The loaded model does not support probability predictions.")
        st.session_state.prediction_results = None
    else:
        cleaned_input = clean_text(user_input, lemmatizer)
        input_vector = vectorizer.transform([cleaned_input])
        probabilities = model.predict_proba(input_vector)[0]
        class_probabilities = list(zip(model.classes_, probabilities))
        sorted_probabilities = sorted(class_probabilities, key=lambda x: x[1], reverse=True)
        st.session_state.prediction_results = sorted_probabilities

if st.session_state.prediction_results:
    sorted_probabilities = st.session_state.prediction_results
    
    st.divider()
    
    top_genre, top_prob = sorted_probabilities[0]
    st.success(f"**Top Predicted Genre: {top_genre.capitalize()}** (Confidence: {top_prob:.2%})")
    
    st.subheader("Other Highly Possible Genres")
    num_other_genres = min(len(sorted_probabilities) - 1, 4)
    
    if num_other_genres > 0:
        cols = st.columns(num_other_genres)
        for i in range(num_other_genres):
            genre, prob = sorted_probabilities[i + 1]
            with cols[i]:
                st.metric(label=genre.capitalize(), value=f"{prob:.2%}")
    else:
        st.info("No other significant genres found.")
        
    st.divider()

    show_probs = st.checkbox("Show full probability breakdown for all genres")

    if show_probs:
        st.subheader("Full Genre Probability Breakdown")
        df = pd.DataFrame(sorted_probabilities, columns=["Genre", "Probability"])
        df['Genre'] = df['Genre'].str.capitalize()
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Genre": "Genre",
                "Probability": st.column_config.ProgressColumn(
                    "Probability",
                    help="The model's confidence in this genre prediction.",
                    format="%.2f%%",
                    min_value=0,
                    max_value=1,
                ),
            },
            hide_index=True,
        )

st.markdown("---")
st.markdown("© 2025 Movie Genre Predictor.")
