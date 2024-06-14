from flask import Flask, render_template, request
import numpy as np
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = request.form['age']
        gender = request.form['gender']
        bmi = request.form['bmi']
        children = request.form['children']
        smoker = request.form['smoker']
        region = request.form['region']
        
        if gender =='male':
            gender = 1
        else:
            gender = 0
        
        if smoker == 'yes':
            smoker = 1
        else:
            smoker = 0
        
        regions = ['northeast', 'northwest', 'southeast', 'southwest']
        for i in regions:
            if region == i:
                region = regions.index(i)
        input_data = [[age, gender, bmi, children, smoker, region]]
        print("Input data for model:", input_data)
        model = joblib.load('./model/insurance_model.pkl')
        prediction = model.predict(input_data)
        
        return render_template('result.html', prediction=round(prediction[0]))
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
