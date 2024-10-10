from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load the trained KMeans model and scaler
kmeans_model = joblib.load('kmeans_clustering_model.pkl')
# scaler = joblib.load('scaler.pkl')

# Define health recommendations based on clusters


def get_health_recommendation(cluster):
    if cluster == 0:
        return "Recommendation for high-risk patients: Immediate intervention required."
    elif cluster == 1:
        return "Recommendation for medium-risk patients: Regular check-ups and monitoring."
    elif cluster == 2:
        return "Recommendation for low-risk patients: Maintain a healthy lifestyle."
    else:
        return "No specific recommendation available for this cluster."

# Define the input preprocessing function


def preprocess_input(data):
    # Expected input fields
    expected_fields = [
        'Age', 'systolic_pressure (mmhg)', 'diastolic_pressure (mmhg)']

    # Convert input data into DataFrame
    input_df = pd.DataFrame([data])

    # Ensure all expected fields are present
    for field in expected_fields:
        if field not in input_df.columns:
            return None, f"Missing required field: {field}"

    # Scaling numeric features
    # numeric_features = [
    #     'Age', 'systolic_pressure (mmhg)', 'diastolic_pressure (mmhg)']
    # input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    return input_df, None


@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    data = request.get_json()

    # Preprocess the input
    processed_data, error = preprocess_input(data)

    if error:
        return jsonify({'error': error}), 400

    # Predict the cluster
    cluster = kmeans_model.predict(processed_data)[0]

    # Get health recommendation
    recommendation = get_health_recommendation(cluster)

    # Return the prediction and recommendation
    return jsonify({
        'predicted_cluster': int(cluster),
        'health_recommendation': recommendation
    })


if __name__ == '__main__':
    app.run(debug=True)
