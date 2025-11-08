from google import genai

client = genai.Client(api_key="AIzaSyA6hAXt9TUq_TwdwNugZKlNfxsYbgHYT_k")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Assuming your dataset is in a CSV file named 'phishing_dataset.csv'
# The dataset should have at least two columns: 'text' (for the email/message content) and 'label' (0 for real, 1 for phishing)
try:
    df = pd.read_csv('phishing_dataset.csv')
except FileNotFoundError:
    print("Error: 'phishing_dataset.csv' not found. Please make sure the dataset file is in the same directory.")
    exit()

# Check if the required columns exist
if 'text' not in df.columns or 'label' not in df.columns:
    print("Error: The dataset must contain 'text' and 'label' columns.")
    exit()

# Split data into training and testing sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_phishing(text):
    # Preprocess the input text
    text = text.lower()  
    # Vectorize 
    text_vec = vectorizer.transform([text]) 
    # Predict
    prediction = model.predict(text_vec)
    return prediction[0]

