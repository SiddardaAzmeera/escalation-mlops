# Escalation Severity Classifier

ML model that predicts whether a customer complaint is HIGH or LOW severity.

## What it does
- Takes any customer complaint text as input
- Returns severity prediction (HIGH/LOW)
- Served as a REST API via FastAPI

## Results
- Accuracy: 93.16%
- Model: Logistic Regression + TF-IDF
- Stack: Python, scikit-learn, FastAPI, uvicorn

## How to run
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Then go to http://localhost:8000/docs

## Built by
Siddarda Azmeera — BTech CS, Amazon CX background
Building toward AI Engineer role.