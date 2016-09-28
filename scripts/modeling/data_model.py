import pandas as pd
import numpy as np
from sklearn import datasets, linear_model

#Would memory always be in GB?

data = pd.read_csv('/Users/ramkishore31/PycharmProjects/workFlow/IPPD/data/ls002_02_25_16.csv')

output = data['SystemTime'] + data['UserTime']

#Remove the %symbol from CPU usage and convert it to integer.
data['PercentCPU'] = [int(i[:-1])for i in data['PercentCPU']]

#Remove G from memory and convert it to integer.
data['mem'] = [int(i[:-1])for i in data['mem']]

data = data.drop('SystemTime', 1)
data = data.drop('UserTime', 1)
data = data.drop('CommandTimed', 1)
data = data.drop('WallTime', 1)
data = data.drop('SampleName', 1)
data = data.drop('trial', 1)

#Use 70 rows for training and the rest for testing.
train_data_input = data[:70]
test_data_input = data[70:]
train_data_output = output[:70]
test_data_output = output[70:]

regr = linear_model.LinearRegression()

regr.fit(train_data_input, train_data_output)

print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(test_data_input) - test_data_output) ** 2))



