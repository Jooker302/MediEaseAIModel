from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models
diabetes_model = joblib.load('diabetes_prediction_model.pkl')
diet_plan_model = joblib.load('dietplan_model.pkl')
exercise_model = joblib.load('exercise_model.pkl')

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make predictions
        predictions = diabetes_model.predict(input_data)
        
        # Return predictions as JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/dietplan', methods=['POST'])
def predict_diet_plan():
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make predictions
        predictions = diet_plan_model.predict(input_data)
        
        # Return predictions as JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/exercise', methods=['POST'])
def predict_exercise():
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make predictions
        predictions = exercise_model.predict(input_data)
        
        # Return predictions as JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
