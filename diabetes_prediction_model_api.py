from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Return predictions as JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
