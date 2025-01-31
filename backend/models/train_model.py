import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("stopwords")
nltk.download("punkt")

# Load dataset
df = pd.read_csv("../dataset/fake_news.csv")  # Adjust the path to your dataset
df = df.dropna()  # Drop missing values

# Text Preprocessing
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

df["cleaned_text"] = df["text"].apply(clean_text)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["label"], test_size=0.2, random_state=42)

# Build Model Pipeline
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", MultinomialNB())
])

# Train Model
model_pipeline.fit(X_train, y_train)

# Save Model
with open("../models/fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model_pipeline, model_file)

print("Model training complete and saved!")
