from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        request.form['gender'],
        float(request.form['ssc_p']),
        float(request.form['hsc_p']),
        float(request.form['degree_p']),
        float(request.form['etest_p']),
        float(request.form['mba_p']),
        request.form['specialisation']
    ]

    # Convert categorical to numeric
    gender = 1 if data[0] == 'Male' else 0
    spec = 1 if data[6] == 'Mkt&Fin' else 0

    input_data = np.array([[gender, data[1], data[2], data[3], data[4], data[5], spec]])
    prediction = model.predict(input_data)[0]

    result = "Placed ✅" if prediction == 1 else "Not Placed ❌"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
