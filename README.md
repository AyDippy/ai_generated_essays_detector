# AI-Generated Essays Detector

![Project Logo/Image - If Applicable](https://unsplash.com/photos/a-computer-generated-image-of-the-letter-a-ZPOoDQc8yMw)

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Welcome to the AI-Generated Essays Detector project! This project is designed to classify essays into two categories: AI-generated and human-written. This README provides an overview of the project, its components, and how to use it.

## Project Overview

- **Goal**: The goal of this project is to develop a machine learning model that can accurately distinguish between AI-generated essays and human-written essays.

- **Data**: The project uses a dataset from Kaggle containing essays as text data and their corresponding classes labeled as "generated."

- **Approach**: The project involves data preprocessing, including text cleaning and lemmatization. It uses Word2Vec embeddings to convert text data into numerical vectors with semantic meaning. A Random Forest Classifier is then trained on the vectorized data to make predictions.

## Dataset

- **Dataset Source**: The dataset used in this project was obtained from Kaggle. You can access the dataset [here](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset).

- **Data Cleaning**: The dataset underwent cleaning, which included removing punctuation, converting text to lowercase, and applying lemmatization to ensure uniformity and enhance model performance.

## Preprocessing

- **Text Preprocessing**: Text data was cleaned using regular expressions to remove punctuation and convert text to lowercase. The WordNet Lemmatizer was applied to further clean and standardize the text.

- **Word2Vec Embeddings**: The Word2Vec model from the Gensim library was used to convert text data into vectors with semantic meaning. These vectors were then averaged to create feature vectors for each essay.

## Model Training

- **Model**: The project uses a Random Forest Classifier for binary classification (AI-generated vs. human-written essays).

- **Accuracy**: The trained model achieved an accuracy of 98 percent on the test data.

## Usage

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/ai-essays-detector.git`

2. Create a virtual environment (recommended): `python -m venv venv`

3. Activate the virtual environment:
   - On macOS and Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`

4. Install the required dependencies: `pip install -r requirements.txt`

5. Run the Streamlit app: `streamlit run app.py`

6. Use the Streamlit web app to input essay text and get predictions. Access the app [here](https://aigeneratedessaysdetector-2gcqrhdm34rnggiltjaghf.streamlit.app/).

## Dependencies

- Python 3.x
- pandas
- numpy
- re (regular expressions)
- gensim (for Word2Vec)
- nltk (Natural Language Toolkit)
- scikit-learn (for machine learning)
- joblib (for model saving/loading)
- streamlit (for the web app)

You can install these dependencies using the `requirements.txt` file provided.

## Contributing

If you would like to contribute to this project, please follow the standard GitHub fork and pull request process. Contributions, bug reports, and feature requests are welcome.

## License

[Provide information about the project's license, if applicable.]

