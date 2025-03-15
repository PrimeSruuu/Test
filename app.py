from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model with error handling
try:
    model_path = 'final_collision_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def home():
    if model is None:
        return "Flask API is running, but the model failed to load", 500
    return "Flask API is working and model is loaded!"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 503
    
    try:
        # Validate that request contains JSON data
        if not request.is_json:
            return jsonify({"error": "Invalid request. JSON data required."}), 400

        data = request.json
        
        # Validate required fields
        required_fields = ['sat1', 'sat2', 'altitude', 'velocity', 'inclination', 
                          'eccentricity', 'raan', 'perigee', 'anomaly']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Extract features from the JSON request with data validation
        try:
            features = np.array([[
                float(data.get('sat1', 0)),
                float(data.get('sat2', 0)),
                float(data.get('altitude', 0)),
                float(data.get('velocity', 0)),
                float(data.get('inclination', 0)),
                float(data.get('eccentricity', 0)),
                float(data.get('raan', 0)),
                float(data.get('perigee', 0)),
                float(data.get('anomaly', 0))
            ]])
        except ValueError as e:
            return jsonify({"error": f"Invalid data format: {str(e)}"}), 400

        # Make prediction
        risk = model.predict(features)[0]
        logger.info(f"Prediction made with risk value: {risk}")

        # Return the result as JSON
        return jsonify({"collision_risk": float(risk)})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    # Don't use debug=True in production
    app.run(host='0.0.0.0', port=5000)