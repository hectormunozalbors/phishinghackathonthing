from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
app = FastAPI()
from  fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class EmailInput(BaseModel):
    text: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

params_file = open('params', 'rb')
params = pickle.load(params_file)
model = LogisticRegression()
model.set_params(**params)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
@app.post("/predict_phishing/")
def predict_phishing(data: EmailInput):
    # Preprocess the input text
    data.text = data.text.lower()  
    # Vectorize 
    
    text_vec = vectorizer.fit_transform([data.text]) 
    # Predict
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0][prediction]
    return {"prediction": prediction, "confidence":probability}

