# Fakenews-detection
# 📰 Fake News Detection with Python

This repository contains a machine learning project that detects whether a piece of news is **REAL** or **FAKE**. It uses natural language processing techniques and a machine learning classifier built using `sklearn`.

## 📌 Project Overview

Fake news, a type of yellow journalism, consists of deliberate misinformation or hoaxes spread via traditional or online social media. The goal of this project is to build a model that can classify news articles as real or fake with high accuracy.

### Technologies Used:
- Python
- Jupyter Lab
- Scikit-learn (`sklearn`)
- Pandas
- NumPy

## 📁 Dataset

The dataset used is `news.csv`, containing 7796 rows and 4 columns:
- `id`: Unique identifier for each news article.
- `title`: Title of the news article.
- `text`: Full text of the news article.
- `label`: Label indicating whether the news is `REAL` or `FAKE`.

Dataset Size: **~29.2MB**

## ⚙️ Project Setup

### Prerequisites

Install the required libraries:

```bash
pip install numpy pandas scikit-learn

🧠 Model Workflow
Import Libraries

Read and Explore the Dataset

Split Data into Training and Testing Sets

Convert Text Data to TF-IDF Features

Train PassiveAggressiveClassifier

Evaluate the Model

🧪 Code Walkthrough
Step 1: Import Dependencies
python
Copy
Edit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

Step 2: Load the Dataset
python
Copy
Edit
df = pd.read_csv('D:\\DataFlair\\news.csv')
print(df.shape)
print(df.head())

Step 3: Split the Data
python
Copy
Edit
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

Step 4: Text Vectorization (TF-IDF)
python
Copy
Edit
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

Step 5: Train Classifier
python
Copy
Edit
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

Step 6: Model Evaluation
python
Copy
Edit
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(conf_matrix)
✅ Output:
Accuracy: ~92.82%

Confusion Matrix:

lua
Copy
Edit
[[589  42]
 [ 49 587]]
📊 Results
True Positives: 589

True Negatives: 587

False Positives: 42

False Negatives: 49

📌 Summary
Implemented a Fake News Detection model using TfidfVectorizer and PassiveAggressiveClassifier.

Achieved high accuracy in detecting fake news articles.

Learned how to process and vectorize text data, and apply an online learning algorithm.

💡 Future Improvements
Use more sophisticated NLP techniques (e.g., word embeddings like Word2Vec or BERT).

Expand dataset to include multiple domains of news.

Deploy the model using Flask or Streamlit for live inference.

📎 Reference
DataFlair - Fake News Detection Project

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
