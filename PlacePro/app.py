from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
model=pickle.load(open('placement_package.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        # Collecting input data from the form
    feature1 = int(request.form.get('feature1'))  # Name of Student
    feature2 = int(request.form.get('feature2'))   # Roll No. (Integer)
    feature3 = int(request.form.get('feature3'))   # No. of DSA Questions (Integer)
    feature4 = float(request.form.get('feature4'))   # CGPA (Float)
    feature5 = float(request.form.get('feature5'))   # Knows ML (Float: 1.0 for Yes, 0.0 for No)
    feature6 = int(request.form.get('feature6'))   # Knows DSA (Integer: 1 for Yes, 0 for No)
    feature7 = float(request.form.get('feature7'))   # Knows Python (Float: 1.0 for Yes, 0.0 for No)
    feature8 = float(request.form.get('feature8'))   # Knows JavaScript (Float: 1.0 for Yes, 0.0 for No)
    feature9 = float(request.form.get('feature9'))   # Knows HTML (Float: 1.0 for Yes, 0.0 for No)
    feature10 = float(request.form.get('feature10')) # Knows CSS (Float: 1.0 for Yes, 0.0 for No)
    feature11 = int(request.form.get('feature11')) # Participated in College Fest (Integer: 1 for Yes, 0 for No)
    feature12 = int(request.form.get('feature12')) # Was in Coding Club (Integer: 1 for Yes, 0 for No)
    feature13 = int(request.form.get('feature13')) # No. of Backlogs (Integer)
    feature14 = int(request.form.get('feature14')) # Age of Candidate (Integer)
    feature15 = int(request.form.get('feature15'))   # Branch of Engineering   
    features = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15]).reshape(1,-1)
        # Make prediction
    prediction = model.predict(features)
        
    return render_template('index.html',prediction_text='Prediction: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
