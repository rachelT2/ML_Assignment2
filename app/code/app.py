#importing the libraries
from flask import Flask,render_template,request
from model.Linear_regression_classes import *;
from model.loaded_model1 import *;
from model.loaded_model2 import *;
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')
car = pd.read_csv('./Cleaned_data.csv')

@app.route('/')
def index():
    return render_template('index.html')              

@app.route('/A1',methods=['GET'])
def A1():
    brand = sorted(car['brand'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    max_power = car['max_power']
    mileage = car['mileage']
    
    return render_template('A1.html',brand=brand, year=year, max_power=max_power, mileage=mileage)


@app.route('/old_predict',methods=['POST'])
def old_predict():
    
    brand_name = request.form.get('brand')
    brand = brand_le.transform([brand_name])

    if not request.form['year']:
        year = year_default
    else:
        year = request.form.get('year')

    if not request.form['max_power']:
        max_power = max_power_default 
    else:
        max_power = float(request.form.get('max_power'))

    if not request.form['mileage']:
        mileage = mileage_default 
    else:
        mileage = float(request.form.get('mileage'))
    
    input_np = np.array([[brand[0],year,max_power,mileage]])
    
    #Scaling the input
    sample = scaler1.transform(input_np)

    #Transforming the scale result to the original value
    predicted_result = np.exp(model1.predict(sample))
    return str(int(predicted_result[0]))





@app.route('/A2',methods=['GET'])
def A2():
    brand = sorted(car['brand'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    max_power = car['max_power']
    mileage = car['mileage']
    
    return render_template('A2.html',brand=brand, year=year, max_power=max_power, mileage=mileage)


@app.route('/new_predict',methods=['POST'])
def new_predict():
    
    brand_name = request.form.get('brand')
    # print(type(brand_name))
    brand = list(brand_ohe.transform([[brand_name]]).toarray()[0])
    # if brand:
    # print(f"{brand_ohe.categories_[0]}")

    if not request.form['year']:
        year = year_default
    else:
        year = request.form.get('year')

    if not request.form['max_power']:
        max_power = max_power_default 
    else:
        max_power = float(request.form.get('max_power'))

    if not request.form['mileage']:
        mileage = mileage_default 
    else:
        mileage = float(request.form.get('mileage'))

    input_np = np.array([[year,max_power,mileage]+brand])
    
    #Scaling the input
    input_np[:, 0:3] = scaler2.transform(input_np[:, 0:3])
    input_np = np.insert(input_np, 0, 1, axis=1)

    #Transforming the scale result to the original value
    # print(f"SHAPE : {input_np}")
    predicted_result = np.exp(model2.predict(input_np.astype(float)))
    return str(int(predicted_result[0]))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

    
