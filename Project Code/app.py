"""
Fetal Health Classification Web Application
This Flask application integrates a pre-trained Random Forest model for fetal health classification.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'optimized_fetal_health_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# Define feature names in the correct order
FEATURE_NAMES = [
    'baseline value',
    'accelerations',
    'fetal_movement',
    'uterine_contractions',
    'light_decelerations',
    'severe_decelerations',
    'prolongued_decelerations',
    'abnormal_short_term_variability',
    'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability',
    'histogram_width',
    'histogram_min',
    'histogram_max',
    'histogram_number_of_peaks',
    'histogram_number_of_zeroes',
    'histogram_mode',
    'histogram_mean',
    'histogram_median',
    'histogram_variance',
    'histogram_tendency'
]

# Class labels for predictions
CLASS_LABELS = {
    1: 'Normal',
    2: 'Suspect',
    3: 'Pathological'
}


@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """
    Render the prediction page.
    """
    if request.method == 'GET':
        return render_template('index.html')
    return predict_result()


@app.route('/contact', methods=['GET'])
def contact():
    """
    Render the contact page.
    """
    return render_template('contact.html')


@app.route('/home')
def home_alias():
    """
    Render the home page (alias for index).
    """
    return render_template('home.html')


@app.route('/inspect', methods=['GET'])
def inspect():
    """
    Render the model information page.
    """
    return render_template('inspect.html')


@app.route('/predict-result', methods=['POST'])
def predict_result():
    """
    Handle prediction requests from the form.
    
    This function:
    1. Retrieves values from the form submission
    2. Validates the input data
    3. Prepares features for the model
    4. Makes a prediction
    5. Calculates confidence
    6. Renders the result page
    """
    
    try:
        # Check if model is loaded
        if model is None:
            return render_template(
                'output.html',
                prediction=0,
                confidence=0,
                error="Model not loaded. Please check server logs."
            ), 500
        
        # Extract values from the form
        input_data = {}
        for feature in FEATURE_NAMES:
            form_key = feature.replace(' ', '_')
            value = request.form.get(form_key)
            
            if value is None or value == '':
                return render_template(
                    'output.html',
                    prediction=0,
                    confidence=0,
                    error=f"Missing value for: {feature}"
                ), 400
            
            try:
                input_data[feature] = float(value)
            except ValueError:
                return render_template(
                    'output.html',
                    prediction=0,
                    confidence=0,
                    error=f"Invalid value for {feature}. Please use numbers."
                ), 400
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get prediction probabilities for confidence score
        try:
            probabilities = model.predict_proba(input_df)[0]
            confidence = max(probabilities) * 100  # Convert to percentage
            confidence = round(confidence, 2)
        except AttributeError:
            # If model doesn't have predict_proba
            confidence = 0
        
        # Get prediction class label
        class_label = CLASS_LABELS.get(int(prediction), 'Unknown')
        
        # Render result page
        return render_template(
            'output.html',
            prediction=int(prediction),
            confidence=confidence,
            class_label=class_label
        )
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template(
            'output.html',
            prediction=0,
            confidence=0,
            error=f"Prediction error: {str(e)}"
        ), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions (returns JSON).
    Useful for programmatic access.
    """
    
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate that all features are present
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Prepare features in correct order
        features = [data[feature] for feature in FEATURE_NAMES]
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get confidence
        try:
            probabilities = model.predict_proba(input_df)[0]
            confidence = float(max(probabilities) * 100)
        except:
            confidence = 0.0
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'class': CLASS_LABELS.get(int(prediction), 'Unknown'),
            'confidence': round(confidence, 2),
            'probabilities': {
                'Normal': float(probabilities[0]) * 100 if len(probabilities) > 0 else 0,
                'Suspect': float(probabilities[1]) * 100 if len(probabilities) > 1 else 0,
                'Pathological': float(probabilities[2]) * 100 if len(probabilities) > 2 else 0
            }
        })
    
    except Exception as e:
        print(f"❌ API Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify server and model status.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(FEATURE_NAMES),
        'classes': list(CLASS_LABELS.values())
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    print(f"❌ Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run the Flask application
    print("="*60)
    print("🏥 Fetal Health Classification Web Application")
    print("="*60)
    print("\n✅ Server starting...")
    print("📍 Navigate to http://127.0.0.1:5000 in your browser")
    print("🔍 Model Information: http://127.0.0.1:5000/inspect")
    print("\nPress CTRL+C to stop the server\n")
    
    # Run on port 5000, accessible from all interfaces
    app.run(debug=True, host='127.0.0.1', port=5000)
