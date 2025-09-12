import pickle

#import the A2 model
loaded_model2 = pickle.load(open('./car_price_predicition2.model','rb'))

# #load the data
# car=pd.read_csv('./Cleaned_data.csv')

model2 = loaded_model2['model']
scaler2= loaded_model2['scaler']
brand_ohe = loaded_model2['brand_ohe']
year_default = loaded_model2['year_default']
max_power_default = loaded_model2['max_power_default']
mileage_default = loaded_model2['mileage_default']