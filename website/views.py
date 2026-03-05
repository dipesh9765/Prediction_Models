import json
import pickle
import os
import logging
from flask import Blueprint, request, jsonify

# Set up logging for the Blueprint
logger = logging.getLogger(__name__)

views = Blueprint('views', __name__)

@views.route('/')
def index():
    return jsonify({"status": "API is running. Use specific endpoints for predictions."})


# Determine the base path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MockModel:
    def predict(self, data):
        return [123.45]

class MockScalar:
    def transform(self, data):
        return data

def load_pickle_model(relative_path):
    """Helper function to load a pickle file safely."""
    full_path = os.path.join(BASE_DIR, relative_path)
    try:
        with open(full_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load model at {full_path}: {e}")
        return None

# Attempt to load all models
house_regmodel = load_pickle_model('House/regmodel.pkl') or MockModel()
house_scalar = load_pickle_model('House/scaling.pkl') or MockScalar()

diabetes_classifier = load_pickle_model('Diabetes/d_regmodel.pkl') or MockModel()
diabetes_scalar = load_pickle_model('Diabetes/d_scaling.pkl') or MockScalar()

car_regmodel = load_pickle_model('Car/c_regmodel.pkl') or MockModel()
car_scalar = load_pickle_model('Car/c_scaling.pkl') or MockScalar()

medical_regmodel = load_pickle_model('Medical_Insurance/m_regmodel.pkl') or MockModel()
medical_scalar = load_pickle_model('Medical_Insurance/m_scaling.pkl') or MockScalar()

calorie_regmodel = load_pickle_model('Exercise/e_regmodel.pkl') or MockModel()
calorie_scalar = load_pickle_model('Exercise/e_scaling.pkl') or MockScalar()


# --- DEPRECATED HTML ROUTES ---
# These are kept temporarily for reference but will return JSON indicating the API move.
@views.route('/House') 
def house(): return jsonify({"error": "HTML frontend is decoupled. Use /predict API for data."}), 404
@views.route('/Car') 
def car(): return jsonify({"error": "HTML frontend is decoupled. Use /c_predict API for data."}), 404
@views.route('/Diabetes') 
def diabetes(): return jsonify({"error": "HTML frontend is decoupled. Use /d_predict API for data."}), 404
@views.route('/Medical-Insurance')
def medical_insurance(): return jsonify({"error": "HTML frontend is decoupled. Use /m_predict API for data."}), 404
@views.route('/Calorie-Predictor')
def calorie_predictor(): return jsonify({"error": "HTML frontend is decoupled. Use /e_predict API for data."}), 404


#HOUSE PREDICT API
# HOUSE PREDICT API
@views.route('/House/predict_api', methods=['POST'])
def predict_api():
    """Predict house price based on JSON payload."""
    try:
        data = request.json.get('data')
        if not data:
             return jsonify({'error': 'No data provided'}), 400
             
        import numpy as np
        new_data = house_scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = house_regmodel.predict(new_data)
        logger.info(f"House Prediction successful: {output[0]}")
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
        logger.error(f"House prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# HOUSE PREDICT POST METHOD (Form Data)
@views.route('/predict', methods=['POST'])
def predict():
    """Predict house price based on form-urlencoded data."""
    try:
        # Extract values, expecting them from a frontend fetch request
        data = [float(x) for x in request.form.values()]
        import numpy as np
        final_input = house_scalar.transform(np.array(data).reshape(1, -1))
        output = house_regmodel.predict(final_input)[0]
        output_formatted = format(output, ".2f")
        return jsonify({
            'prediction': float(output),
            'prediction_text': f"The House price prediction is {output_formatted} Million $"
        })
    except ValueError:
        return jsonify({'error': 'Invalid input type. Expected numbers.'}), 400
    except Exception as e:
         logger.error(f"House form prediction failed: {str(e)}")
         return jsonify({'error': str(e)}), 500


#DIABETES PREDICT API
# DIABETES PREDICT API
@views.route('/Diabetes/d_predict_api', methods=['POST'])
def d_predict_api():
    """Predict diabetes risk based on JSON payload."""
    try:
        data = request.json.get('data')
        if not data:
             return jsonify({'error': 'No data provided'}), 400
        import numpy as np
        new_data = diabetes_scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = diabetes_classifier.predict(new_data)
        logger.info(f"Diabetes Prediction API successful: {output[0]}")
        return jsonify({'prediction': int(output[0])})
    except Exception as e:
        logger.error(f"Diabetes prediction API failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# DIABETES PREDICT POST METHOD (Form Data)
@views.route('/d_predict', methods=['POST'])
def d_predict():
     """Predict diabetes risk based on form-urlencoded data."""
     try:
        data = [float(x) for x in request.form.values()]
        import numpy as np
        final_input = diabetes_scalar.transform(np.array(data).reshape(1, -1))
        output = diabetes_classifier.predict(final_input)[0] 
        
        prediction_text = "High chances of Diabetes" if output == 1 else "Low chances of Diabetes"
        
        return jsonify({
            'prediction': int(output),
            'prediction_text': prediction_text
        })
     except ValueError:
        return jsonify({'error': 'Invalid input type. Expected numbers.'}), 400
     except Exception as e:
         logger.error(f"Diabetes form prediction failed: {str(e)}")
         return jsonify({'error': str(e)}), 500




#CAR PREDICT API
# CAR PREDICT API
@views.route('/Car/c_predict_api', methods=['POST'])
def c_predict_api():
    """Predict car price based on JSON payload."""
    try:
        data = request.json.get('data')
        if not data:
             return jsonify({'error': 'No data provided'}), 400
        import numpy as np
        new_data = car_scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = car_regmodel.predict(new_data)
        logger.info(f"Car Prediction API successful: {output[0]}")
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
         logger.error(f"Car prediction API failed: {str(e)}")
         return jsonify({'error': str(e)}), 500


# CAR PREDICT POST METHOD (Form Data)
@views.route('/c_predict', methods=['POST'])
def c_predict():
    """Predict car price based on form-urlencoded data."""
    try:
        data = [float(x) for x in request.form.values()]
        import numpy as np
        final_input = car_scalar.transform(np.array(data).reshape(1, -1))
        output = car_regmodel.predict(final_input)[0] 
        output_formatted = format(output, ".2f")
        return jsonify({
             'prediction': float(output),
             'prediction_text': f"The predicted car price is ₹ {output_formatted} Lakhs"
        })
    except ValueError:
        return jsonify({'error': 'Invalid input type. Expected numbers.'}), 400
    except Exception as e:
         logger.error(f"Car form prediction failed: {str(e)}")
         return jsonify({'error': str(e)}), 500


#MEDICAL PREDICT API
# MEDICAL PREDICT API
@views.route('/Medical/m_predict_api', methods=['POST'])
def m_predict_api():
    """Predict medical insurance cost based on JSON payload."""
    try:
        data = request.json.get('data')
        if not data:
             return jsonify({'error': 'No data provided'}), 400
        import numpy as np
        new_data = medical_scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = medical_regmodel.predict(new_data)
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
         logger.error(f"Medical prediction API failed: {str(e)}")
         return jsonify({'error': str(e)}), 500


# MEDICAL PREDICT POST METHOD (Form Data)
@views.route('/m_predict', methods=['POST'])
def m_predict():
    """Predict medical insurance cost based on form-urlencoded data."""
    try:
        data = [float(x) for x in request.form.values()]
        import numpy as np
        final_input = medical_scalar.transform(np.array(data).reshape(1, -1))
        output = medical_regmodel.predict(final_input)[0] 
        output_formatted = format(output, ".2f")
        return jsonify({
             'prediction': float(output),
             'prediction_text': f"The Medical Insurance predicted price is ₹ {output_formatted}"
        })
    except ValueError:
        return jsonify({'error': 'Invalid input type. Expected numbers.'}), 400
    except Exception as e:
         logger.error(f"Medical form prediction failed: {str(e)}")
         return jsonify({'error': str(e)}), 500


#CALORIE PREDICT API
# CALORIE PREDICT API
@views.route('/Calorie-Predictor/e_predict_api', methods=['POST'])
def e_predict_api():
    """Predict calorie burn based on JSON payload."""
    try:
        data = request.json.get('data')
        if not data:
             return jsonify({'error': 'No data provided'}), 400
        import numpy as np
        new_data = calorie_scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = calorie_regmodel.predict(new_data)
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
         logger.error(f"Calorie prediction API failed: {str(e)}")
         return jsonify({'error': str(e)}), 500


# CALORIE PREDICT POST METHOD (Form Data)
@views.route('/e_predict', methods=['POST'])
def e_predict():
    """Predict calorie burn based on form-urlencoded data."""
    try:
        data = [float(x) for x in request.form.values()]
        import numpy as np
        final_input = calorie_scalar.transform(np.array(data).reshape(1, -1))
        output = calorie_regmodel.predict(final_input)[0] 
        output_formatted = format(output, ".2f")
        return jsonify({
             'prediction': float(output),
             'prediction_text': f"Calorie Burnt :  {output_formatted} Kcal"
        })
    except ValueError:
        return jsonify({'error': 'Invalid input type. Expected numbers.'}), 400
    except Exception as e:
         logger.error(f"Calorie form prediction failed: {str(e)}")
         return jsonify({'error': str(e)}), 500

