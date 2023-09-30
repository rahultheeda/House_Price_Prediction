import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('data.csv')
data.columns
data.head()

#missing values
data.fillna(0,inplace=True)
#encoding
label_encoder = LabelEncoder()
colm_to_encode=['date','bedrooms','bathrooms','street', 'city','statezip','country']
for col in colm_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

#split data into x and y
X=data.drop(columns=['price'])
y=data['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50,random_state=10)
#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#training model
model=LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
coeff_df=pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_df

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Abosulte Error is {mae}")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

#model is ready for new data
new_data = pd.DataFrame({'date': ['2014-05-02 00:00:00'],
                         'bedrooms': [3.0],
                         'bathrooms': [1.50],
                         'sqft_living': [1340],
                         'sqft_lot': [7912],
                         'floors': [1.5],
                         'waterfront': [0],
                         'view': [0],'condition': [3],
                         'sqft_above': [1340],
                         'sqft_basement':[0],
                         'yr_built':[1955],
                         'yr_renovated':[2005],
                         'street':['18810 Densmore Ave N'],
                         'city':['Shoreline'],
                         'statezip':['WA 98133'],
                        'country':[' USA']})
colm_to_encode=['date', 'yr_built', 'yr_renovated', 'street', 'city','statezip', 'country']
for col in colm_to_encode:
    new_data[col] = label_encoder.fit_transform(new_data[col])
    
new_data_scaled=scaler.transform(new_data)

predicted_price = model.predict(new_data_scaled)
print(f"predicted price: {predicted_price[0]}")