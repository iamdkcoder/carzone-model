from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
model=pickle.load(open('RandomForestRegressionModel.pkl','rb'))
car=pd.read_csv('cleaned_car.csv')

@app.route('/')
def index():
    manufacturer=sorted(car['Manufacturer'].unique())
    model=sorted(car['Model'].unique())
    year=sorted(car['Year'].unique(),reverse=True)
    fuel_type=sorted(car['Fuel_Type'].unique())
    owner_type = car['Owner_Type'].unique()
    return render_template('index.html',Manufacturer=manufacturer, Models=model, Years=year, Fuel_Types=fuel_type,Owner_Type=owner_type)

@app.route('/predict',methods=['POST'])
def predict():

    company=request.form.get('Manufacturer')
    car_model=request.form.get('Model')
    year=int(request.form.get('Year'))
    fuel_type=request.form.get('Fuel-Type')
    driven=int(request.form.get('Kms_Driven'))
    owner=request.form.get('Owner-Type')
    mileage=float(request.form.get('Mileage'))
    engine=float(request.form.get('Engine'))
    power=float(request.form.get('Power'))
    # print(company,car_model,year,driven,fuel_type,owner,mileage,engine,power)
    # return ""

    prediction=model.predict(pd.DataFrame([[company,car_model,year,driven,fuel_type,owner,mileage,engine,power]],columns=['Manufacturer','Model','Year','Kilometers_Driven','Fuel_Type','Owner_Type','Mileage','Engine','Power']
                              ))
    # print(prediction)

    return str(np.round(prediction[0],2))
if __name__=='__main__':
    app.run(debug=True)