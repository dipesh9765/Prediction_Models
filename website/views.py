import json
import pickle

from flask import Flask,Blueprint, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

views =Blueprint('views', __name__)


@views.route('/')
def index():
    return render_template('base.html')



house_regmodel = pickle.load(open('./website/House/regmodel.pkl', 'rb'))
house_scalar = pickle.load(open('./website/House/scaling.pkl', 'rb'))

diabetes_classifier = pickle.load(open('./website/Diabetes/d_regmodel.pkl','rb'))
diabetes_scalar = pickle.load(open('./website/Diabetes/d_scaling.pkl', 'rb'))



@views.route('/House') 
def house():
    return render_template('house.html')

@views.route('/Car') 
def car():
    return render_template('car.html')
@views.route('/Diabetes') 
def diabetes():
    return render_template('diabetes.html')



#HOUSE PREDICT API
@views.route('/House/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = house_scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = house_regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


#DIABETES PREDICT API
@views.route('/Diabetes/d_predict_api', methods=['POST'])
def d_predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = diabetes_scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = diabetes_classifier.predict(new_data)
    print(output[0])
    return jsonify(output[0])


# DIABETED PREDICT POST METHOD
@views.route('/d_predict', methods=['POST'])
def d_predict():
    data = [float(x) for x in request.form.values()]
    final_input = diabetes_scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = diabetes_classifier.predict(final_input)[0] 
    print(output)
    if output == 1:
        return render_template("diabetes.html",prediction_text=" You have Diabetes" )
    else:
        return render_template("diabetes.html", prediction_text=" You don't have Diabetes" )

# HOUSE PREDICT POST METHOD
@views.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = house_scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = house_regmodel.predict(final_input)[0]
    output = format(output,".2f")
    return render_template("house.html", prediction_text="The House price prediction is {} Million $".format(output))

