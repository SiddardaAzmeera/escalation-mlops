from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict(review: ReviewInput):
    vec = vectorizer.transform()
    # ADDED [0] HERE TO PREVENT THE 500 ERROR
    prediction = model.predict(vec)[0]
    severity = "HIGH" if prediction == 1 else "LOW"
    return {"severity": severity, "input": review.text}

@app.get("/", response_class=HTMLResponse)
def home():
    return 
