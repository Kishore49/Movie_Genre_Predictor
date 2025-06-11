import streamlit as st
import re
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬", layout="wide")

@st.cache_resource
def load_artifacts():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

# Text preprocessing
def clean_text(text, lemmatizer):
    text = re.sub(r'http\S+|[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

st.markdown("""
    <style>
    .main {background-color: #f5f6fa;}
    .stButton>button {background-color: #4F8BF9; color: white;}
    .stTextArea textarea {background-color: black; color: white;}
    </style>
""", unsafe_allow_html=True)

# Load resources
try:
    vectorizer, model = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are available.")
    st.stop()

# App UI
st.title("🎬 Movie Genre Predictor")
st.markdown("Predict the genre of a movie based on its plot synopsis.")

user_input = st.text_area(
    "Enter the movie synopsis:",
    height=200,
    placeholder="A young wizard discovers his magical heritage..."
)

show_probs = st.checkbox("Show genre probabilities")

if st.button("Predict Genre"):
    if user_input.strip():
        cleaned_input = clean_text(user_input, lemmatizer)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]
        st.success(f"**Predicted Genre:** {prediction.capitalize()}")

        if show_probs and hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_vector)[0]
            st.subheader("Genre Probabilities")
            st.dataframe({
                "Genre": model.classes_,
                "Probability": np.round(proba, 3)
            })
    else:
        st.warning("Please enter a synopsis before predicting.")

st.markdown("---")
st.markdown("© 2025 Movie Genre Predictor.")
