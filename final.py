import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('/Users/yashnegi/Desktop/symp_check.pkl')
vectorizer = joblib.load('/Users/yashnegi/Desktop/vectorizer.pkl')

@app.route('/')
def home():
    return "Welcome to the Symptoms Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    # Transform the input symptoms using the vectorizer
    symptoms_vec = vectorizer.transform([symptoms])

    # Make a prediction
    prediction = model.predict(symptoms_vec)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

@app.route('/predict', methods=['GET'])
def predict_get():
    return "Use POST method to make predictions"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test successful'})

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)