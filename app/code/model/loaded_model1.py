import pickle
#import the model
loaded_model1 = pickle.load(open('./car_price_predicition1.model','rb'))

# #load the data
# car=pd.read_csv('./Cleaned_data.csv')

model1 = loaded_model1['model']
scaler1= loaded_model1['scaler']
brand_le = loaded_model1['brand_le']
year_default = loaded_model1['year_default']
max_power_default = loaded_model1['max_power_default']
mileage_default = loaded_model1['mileage_default']