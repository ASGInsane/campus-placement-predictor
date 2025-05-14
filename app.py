from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Example: Collect form data (adjust based on actual inputs)
    cgpa = float(request.form['cgpa'])
    # Add other features here...
    features = np.array([[cgpa]])  # Replace with full feature list

    prediction = model.predict(features)[0]
    
    # For chart
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        prob_data = {'Placed': round(probs[1]*100, 2), 'Not Placed': round(probs[0]*100, 2)}
    else:
        prob_data = None

    return render_template('index.html', prediction=prediction, probabilities=prob_data)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)


