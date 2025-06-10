# 🎬 Movie Genre Prediction

This project aims to predict the **genre of movies** based on their **plot synopsis** using machine learning and NLP techniques. Given a movie's description, the model classifies it into one of the known genres like drama, comedy, action, etc.

---


### 1. Clone the repository

```bash
git clone 
cd movie-genre-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # For Windows
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 📄 Data

* **train.csv**: Contains movie synopses and their corresponding genres for model training.
* **test.csv**: Contains movie synopses without genre labels for inference.

---

## 🧹 Preprocessing

The text data undergoes the following steps:

* Removal of URLs and non-alphabet characters
* Lowercasing all text
* Removing extra spaces
* Lemmatization using `WordNetLemmatizer`
* (Optional) Stopword removal (if extended)

---

## Model Training

* The **TF-IDF Vectorizer** is used to transform plot synopses into numerical vectors.
* **Multinomial Naive Bayes** classifier is trained using these features.
* Optimal hyperparameters are selected via **GridSearchCV** (best alpha = `2.0`).

Artifacts saved after training:

* `model.pkl`: Trained Naive Bayes classifier
* `vectorizer.pkl`: Fitted TF-IDF vectorizer

---

## 🧪 Evaluation

* Accuracy and classification report are used to evaluate performance.
* Cross-validation ensures generalization to unseen data.

---

## 📝 How to Run

### 1. Train the Model

Ensure `train.csv` is in the `Movie_genre_prediction/` directory, then run:

```bash
python train_model.py
```

This will generate the `model.pkl` and `vectorizer.pkl` files.

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

Then enter a movie synopsis in the UI to get a predicted genre.

---

## 📊 Streamlit App Features

* Clean user interface to input a synopsis
* Predicts the movie genre
* Option to view the probability distribution across all genres

---

## 🔍 Hyperparameter Tuning

* GridSearchCV is used on the Naive Bayes `alpha` parameter.
* The optimal value (`alpha=2.0`) yielded the best cross-validation results.

---

## 🏁 Results

* The trained model performs well on the dataset.
* Final predictions for test data can be exported using the notebook or evaluation scripts.

---


