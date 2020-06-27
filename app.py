import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
app = Flask(__name__)
model = joblib.load('Students_mark_predictor_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0][0].round(2)
    return render_template('index.html', prediction_text="Marks obtain by student should be {}" .format(output))

if __name__=='__main__':
    app.run(debug=True)
