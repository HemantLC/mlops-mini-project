from flask import Flask, render_template, request
import mlflow

from preprocessing_utility import normalize_text
import pickle

import dagshub

mlflow.set_tracking_uri("https://dagshub.com/HemantLC/mlops-mini-project.mlflow")
dagshub.init(repo_owner='HemantLC', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)

# Load the BOW vectoriser
vectoriser = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Load model from model registry
model_name = 'my_model'
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html', result = None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # clean
    text = normalize_text(text)
    print(text)
    print('***************************')

    # BOW
    features = vectoriser.transform([text])
    print(features)
    print('**************************')
    


    # Prediction
    result = model.predict(features)
    print(result)
    print('**************************')
    

    # Show
    return render_template('index.html', result= result[0])

app.run(debug=True, port=8000)