#importing the necessary libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
import nltk

# Downloading wordnet for data cleaning
nltk.download('wordnet')

# Loading the random forest classifier model
model = joblib.load('AI_generated_essay_detector_model.pkl')

#loading word2vec trained
word2vec = joblib.load('word2vec_model.pkl')

# Streamlit webpage title
st.title("AI Generated Essay Detector")

# Text box for user input
user_input = st.text_area("Enter the essay text here", "")

# Preprocessing function
def preprocess_text(text):
    # Lemmatizer
    wl = WordNetLemmatizer()

    # Cleaning the text
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [wl.lemmatize(word) for word in review]
    review = ' '.join(review)

    # Preprocessing for Word2Vec
    words = simple_preprocess(review)

    # Average Word2Vec
    if len(words) == 0:
        return np.zeros((100,))  # Adjust the size to match your model's input
    else:
        return np.mean([word2vec.wv[word] for word in words if word in word2vec.wv.index_to_key], axis=0)

def predict(text):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    # Reshape to match the input format of the model
    processed_text = pd.DataFrame(np.array(processed_text).reshape(1,-1))
    # Make prediction
    prediction = model.predict(processed_text)
    return prediction

# Predict button
if st.button("Predict"):
    prediction = predict(user_input)
    if prediction == 0:
        st.success("Prediction: The essay is Human-written")
    else:
        st.success("Prediction: The essay is AI-generated")
