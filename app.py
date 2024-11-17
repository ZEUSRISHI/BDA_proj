from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('count_vectorizer1.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define the route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        text_vector = vectorizer.transform([user_input])  # Vectorize the input text
        prediction = lr_model.predict(text_vector)  # Predict using the model
        proba = lr_model.predict_proba(text_vector)  # Get probabilities
        sentiment = "Positive" if prediction[0] == 1 else "Negative"  # Adjust this based on your labeling
        
        return jsonify({
            'sentiment': sentiment,
            'probability': f"{proba.max() * 100:.2f}%",
            'input_text': user_input
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
