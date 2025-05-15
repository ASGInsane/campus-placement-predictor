from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Mapping for college names
college_map = {
    'IIT Bombay': 0,
    'IIT Delhi': 1,
    'NIT Trichy': 2,
    'BITS Pilani': 3,
    'VIT Vellore': 4,
    'SRM University': 5
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        cgpa = float(request.form['cgpa'])
        gender = request.form['gender']
        tenth = float(request.form['tenth'])
        twelfth = float(request.form['twelfth'])
        backlogs = int(request.form['backlogs'])
        degree = float(request.form['degree'])
        college = request.form['college']

        # Convert categorical values
        gender_num = 1 if gender == 'Male' else 0
        college_num = college_map.get(college, -1)

        # Combine into features array (7 features only)
        features = np.array([[cgpa, gender_num, tenth, twelfth, backlogs, degree, college_num]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Get probability if available
        prob_data = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            prob_data = {
                'Placed': round(probs[1] * 100, 2),
                'Not Placed': round(probs[0] * 100, 2)
            }

        return render_template('index.html', prediction=prediction, probabilities=prob_data)

    except Exception as e:
        return f"Error occurred: {e}", 500

@app.route('/dashboard')
def dashboard():
    # Example stats for dashboard; replace with real data or model insights
    stats = {
        'total_students': 150,
        'placed_students': 120,
        'not_placed_students': 30,
        'placement_rate': round(120 / 150 * 100, 2),
        'top_college': 'IIT Bombay',
        'average_cgpa': 8.5,
    }
    return render_template('dashboard.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)





