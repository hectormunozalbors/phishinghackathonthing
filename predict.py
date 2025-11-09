from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
app = FastAPI()
from  fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import numpy as np

class EmailInput(BaseModel):
    text: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("phish_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("phish_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.post("/predict_phishing/")
def predict_phishing(data: EmailInput):
    # Preprocess input text
    text = data.text.lower()
    text_vec = vectorizer.transform([text])

    # Make prediction
    raw_pred = model.predict(text_vec)[0]
    raw_probs = model.predict_proba(text_vec)[0]

    # Convert numpy scalars to Python native types
    pred = int(raw_pred.item() if isinstance(raw_pred, np.generic) else raw_pred)

    if isinstance(raw_probs, np.ndarray) and raw_probs.ndim > 0:
        conf = float(raw_probs[int(pred)].item())
    else:
        conf = float(raw_probs.item() if isinstance(raw_probs, np.generic) else raw_probs)

    # Encode safely to JSON-serializable form
    response = jsonable_encoder({
        "prediction": pred,
        "confidence": conf
    })

    print("DEBUG: returning", response)
    return response


