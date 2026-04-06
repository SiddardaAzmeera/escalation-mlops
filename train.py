import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("Loading data...")
data = fetch_20newsgroups(subset='all')
texts = data.data
labels = [1 if t in [1,2,3,4] else 0 for t in data.target]

print("Training model...")
vec = TfidfVectorizer(max_features=5000)
X = vec.fit_transform(texts)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.2%}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)

print("Model Successfully Saved Sir!")