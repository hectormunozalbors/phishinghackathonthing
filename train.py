import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    """
    Main function to train and evaluate the email phishing detection model.
    """
    # Load the email dataset
    try:
        df = pd.read_csv('phishing_email.csv')
    except FileNotFoundError:
        print("Error: 'phishing_email.csv' not found. Please make sure the dataset file is in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Check for expected columns and handle missing 'Email Text'
    if 'Email Text' not in df.columns or 'Email Type' not in df.columns:
        print(f"Error: The dataset must contain 'text_combined' and 'label' columns. Found columns: {df.columns.tolist()}")
        return
    
    # Drop rows where 'Email Text' is missing, as they cannot be used for training.
    df.dropna(subset=['Email Text'], inplace=True)

    # Rename columns for consistency
    df = df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})

    # Convert 'Phishing Email'/'Safe Email' labels to 1/0
    df['label'] = df['label'].apply(lambda x: 1 if x == 'Phishing Email' else 0)

    # Split data into training and testing sets
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n--- Training Model and Showing Accuracy Progression ---")
    # We will train the model on increasing chunks of data to see how accuracy improves.
    num_steps = 10
    accuracies = []
    for i in range(1, num_steps + 1):
        subset_size = int((i / num_steps) * X_train_vec.shape[0])
        if subset_size == 0: continue

        # Train a new model on the current subset of training data
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec[:subset_size], y_train.iloc[:subset_size])
        
        # Evaluate the model on the entire test set
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Training with {subset_size} samples ({i * 10}% of data)... Test Accuracy: {accuracy:.3f}")

    # --- Test the prediction function ---
    test_emails = [
        "URGENT: Your account has been suspended. Click here to verify.",
        "Hi team, please find the attached report for our quarterly review.",
        "Congratulations! You've won a prize. Please provide your details to claim $1,000,000."
    ]

    # Final evaluation with the model trained on all data
    print("\n--- Final Model Evaluation ---")
    print(f"Final Model Accuracy: {accuracies[-1]:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\n--- Testing with sample emails ---")
    for email in test_emails:
        prediction, probability = predict_phishing(email, vectorizer, model)
        result = "Phishing" if prediction == 1 else "Safe Email"
        print(f"Email: '{email[:50]}...' -> Prediction: {result} (Confidence: {probability*100:.2f}%)")

if __name__ == "__main__":
    main()
