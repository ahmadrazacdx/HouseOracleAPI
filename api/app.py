from flask import Flask, jsonify, request
from api.prediction import get_price_prediction, recommend_properties
from api.utils import (
    create_df,
    make_recommendation_df,
    log_transform,
    exp_transform
)
import traceback
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Welcome to the HouseOracle API'


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for property price prediction
    Expects JSON payload with property features and purpose (rent/sale)
    Returns JSON with predicted price or error message
    """
    """
    Single endpoint for:
          1) Property price prediction
          2) Returning top-3 recommended properties (with converted units)
        Expects JSON payload with property features and purpose ("rent" or "sale").
        Returns JSON with:
          - predicted_price: float
          - recommendations: list of 3 property dicts
        """
    try:
        data = request.get_json()
        if not data or 'purpose' not in data:
            return jsonify({'error': 'Missing required fields'}), 400

        purpose = data['purpose'].lower()
        if purpose not in ('rent', 'sale'):
            return jsonify({'error': 'Invalid purpose. Use "rent" or "sale"'}), 400

        del data['purpose']
        predicted_price = get_price_prediction(data, purpose)
        input_df = create_df(data)
        input_df = make_recommendation_df(input_df, predicted_price)
        recommendations = recommend_properties(input_df, purpose)

        recommendations_json = recommendations.to_dict(orient='records')

        return jsonify({
            'predicted_price': predicted_price,
            'recommendations': recommendations_json
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
