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
    # Collecting form data
    cgpa = float(request.form['cgpa'])
    college = request.form['college']
    interview_score = float(request.form['interview_score'])
    years_of_experience = float(request.form['years_of_experience'])

    # Convert categorical features (e.g., college) to numerical format
    # Map colleges to numerical values (add more mapping if needed)
    college_mapping = {
        'IIT Bombay': 1,
        'IIT Delhi': 2,
        'NIT Trichy': 3,
        'BITS Pilani': 4,
        'VIT Vellore': 5,
        'SRM University': 6
    }
    college_value = college_mapping.get(college, 0)

    # Create the feature array with the necessary features
    features = np.array([[cgpa, college_value, interview_score, years_of_experience]])

    # Predicting the placement status
    prediction = model.predict(features)[0]

    # For probability chart
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        prob_data = {'Placed': round(probs[1]*100, 2), 'Not Placed': round(probs[0]*100, 2)}
    else:
        prob_data = None

    return render_template('index.html', prediction=prediction, probabilities=prob_data)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)

