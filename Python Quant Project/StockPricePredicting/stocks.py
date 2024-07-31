# using neural networks lSTM to predict from 60days of information
# to predict the next day

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
#needed to add the .python. in order for the code to not highlight for tensor
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTMV1
 
#load data
company = 'FB'
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)
data = web.DataReader(company, 'yahoo', start, end)

#prepareData
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
# how far back to look
predicition_days = 60

x_train = []
y_train = []

for x in range(predicition_days, len(scaled_data)):
    x_train.append(scaled_data[x-predicition_days:x,0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#build the model
model = Sequential()
# numbers change up depending on the performance, more layers longer to train
model.add(LSTMV1(units = 50, return_sequences =True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTMV1(units =50, return_sequences =True))
model.add(Dropout(0.2))
model.add(LSTMV1(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #predict the closing value

model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs= 25, batchsize =32)

'''TEST the mmodel accuracy on Existing Data'''
# load test Data
test_start = dt.datetime(2022,1,1)
test_end = dt.datetime.now()
test_data = web.DataReader (company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis =0)

model_inputs = total_dataset[len(total_dataset) -len(test_data) -predicition_days:].value

model_inputs = model_inputs.reshape(-1,1)

model_inputs = scaler.transform(model_inputs)

# make some predictions on test data
x_test = []
for x in range(prediciton_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

predicted_prices = model.predict(x_test)
predicted_pries = scaler.inverse_transform(predicted_prices)

#plot the predictions
plt.plot(actual_prices,color = "black", label = f"Actual {company} Price")
plt.plot(predictec_prices, color = 'green', label = f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()