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
    gender = request.form['gender']
    tenth = float(request.form['tenth'])
    twelfth = float(request.form['twelfth'])
    backlogs = int(request.form['backlogs'])
    degree = float(request.form['degree'])
    college = request.form['college']

    # Convert categorical features (e.g., gender, college) to numerical format
    gender = 1 if gender == 'Male' else 0  # Assuming Male=1 and Female=0
    college = 1 if college == 'College A' else 0  # Example; you may want to map to actual colleges

    # Create the feature array with all 7 features
    features = np.array([[cgpa, gender, tenth, twelfth, backlogs, degree, college]])

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
