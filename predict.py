from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
import pickle
app = FastAPI()

@app.post("/load_model/")
def load_model():
    params_file = open('model', 'rb')
    params = pickle.load(params_file)
    global model
    model = LogisticRegression()
    model.set_params(params)

@app.post("/predict_phishing/")
def predict_phishing(text, vectorizer):
    # Preprocess the input text
    text = text.lower()  
    # Vectorize 
    text_vec = vectorizer.transform([text]) 
    # Predict
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0][prediction]
    return prediction, probability

