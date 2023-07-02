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


car_regmodel = pickle.load(open('./website/Car/c_regmodel.pkl','rb'))
car_scalar = pickle.load(open('./website/Car/c_scaling.pkl', 'rb'))

medical_regmodel = pickle.load(open('./website/Medical_Insurance/m_regmodel.pkl','rb'))
medical_scalar = pickle.load(open('./website/Medical_Insurance/m_scaling.pkl', 'rb'))

calorie_regmodel = pickle.load(open('./website/Exercise/e_regmodel.pkl','rb'))
calorie_scalar = pickle.load(open('./website/Exercise/e_scaling.pkl', 'rb'))



@views.route('/House') 
def house():
    return render_template('house.html')

@views.route('/Car') 
def car():
    return render_template('car.html')

@views.route('/Diabetes') 
def diabetes():
    return render_template('diabetes.html')

@views.route('/Medical-Insurance')
def medical_insurance():
    return render_template('medical.html')

@views.route('/Calorie-Predictor')
def calorie_predictor():
    return render_template('exercise.html')


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

# HOUSE PREDICT POST METHOD
@views.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = house_scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = house_regmodel.predict(final_input)[0]
    output = format(output,".2f")
    return render_template("house.html", prediction_text="The House price prediction is {} Million $".format(output))


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




#CAR PREDICT API
@views.route('/Car/c_predict_api', methods=['POST'])
def c_predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = car_scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = car_regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


#CAR PREDICT POST METHOD
@views.route('/c_predict', methods=['POST'])
def c_predict():
    data = [float(x) for x in request.form.values()]
    final_input = car_scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = car_regmodel.predict(final_input)[0] 
    print(output)
    output = format(output,".2f")
    return render_template("car.html", prediction_text="The Car price prediction is ₹ {} Lakhs".format(output))


#MEDICAL PREDICT API
@views.route('/Medical/m_predict_api', methods=['POST'])
def m_predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = medical_scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = medical_regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


#MEDICAL PREDICT POST METHOD
@views.route('/m_predict', methods=['POST'])
def m_predict():
    data = [float(x) for x in request.form.values()]
    final_input = medical_scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = medical_regmodel.predict(final_input)[0] 
    print(output) 
    output = format(output,".2f")
    return render_template("medical.html", prediction_text="The Medical Insurance predicted  price is ₹ {}".format(output))


#CALORIE PREDICT API
@views.route('/Calorie-Predictor/e_predict_api', methods=['POST'])
def e_predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = calorie_scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = calorie_regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


#CALORIE PREDICT POST METHOD
@views.route('/e_predict', methods=['POST'])
def e_predict():
    data = [float(x) for x in request.form.values()]
    final_input = calorie_scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = calorie_regmodel.predict(final_input)[0] 
    print(output) 
    output = format(output,".2f")
    return render_template("exercise.html", prediction_text="Calorie Burnt :  {} Kcal".format(output))

