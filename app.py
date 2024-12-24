from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('chronic_disease_prediction_model.h5')

# Load and preprocess sample data to get the scaler
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')

# Initialize the scaler and fit it with the original data
scaler = StandardScaler()
scaler.fit(X)

@app.route('/')
def home():
    return render_template('index.html')  # Make sure you create an index.html file for the home page

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    patient_name = request.form.get('name')  # Get the patient's name
    input_features = [float(x) for x in request.form.values() if x != patient_name]  # Get the input features
    final_features = [np.array(input_features)]
    
    # Scale the input features using the previously fitted scaler
    scaled_features = scaler.transform(final_features)

    # Predict using the loaded model
    prediction = model.predict(scaled_features)

    # Output prediction result
    output = (prediction > 0.5).astype("int32")  # Convert the probability to binary output (0 or 1)
    result = 'Positive' if output == 1 else 'Negative'

    return render_template('index.html', prediction_text=f'Chronic Disease Prediction for {patient_name}: {result}')

if __name__ == "__main__":
    app.run(debug=True)
