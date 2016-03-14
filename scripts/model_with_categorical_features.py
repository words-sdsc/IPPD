import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.feature_extraction import DictVectorizer


#Would memory always be in GB?

data = pd.read_csv('/Users/ramkishore31/PycharmProjects/workFlow/IPPD/data/ls002_02_25_16.csv')

output = data['SystemTime'] + data['UserTime']

#Remove the %symbol from CPU usage and convert it to integer.
data['PercentCPU'] = [int(i[:-1])for i in data['PercentCPU']]

#Remove G from memory and convert it to integer.
data['mem'] = [int(i[:-1])for i in data['mem']]
#Convert the ppn feature to sting
data['ppn'] = [str(i) for i in data['ppn']]
data['ExitStatus'] = [str(i) for i in data['ExitStatus']]

data = data.drop('SystemTime', 1)
data = data.drop('UserTime', 1)
data = data.drop('CommandTimed', 1)
data = data.drop('WallTime', 1)
data = data.drop('SampleName', 1)
data = data.drop('trial', 1)

data2 = data.to_dict(orient='records')
vec = DictVectorizer()
data2 = np.array(vec.fit_transform(data2).toarray())

#Use 70 rows for training and the rest for testing.
train_data_input = data2[:70]
test_data_input = data2[70:]
train_data_output = output[:70]
test_data_output = output[70:]

regr = linear_model.LinearRegression()

regr.fit(train_data_input, train_data_output)
print "ppn as categorical"
print("Root mean square error: %.2f"
      % np.sqrt(np.mean((regr.predict(test_data_input) - test_data_output) ** 2)))
test_data_output = np.array(test_data_output)
predicted_output = (regr.predict(test_data_input))
error = 0
for i in range(len(test_data_output)):
      error += np.abs(predicted_output[i] - test_data_output[i])

print "Mean absolute error",error / float(len(test_data_output))

#Use 70 rows for training and the rest for testing.
train_data_input = data[:70]
test_data_input = data[70:]
train_data_output = output[:70]
test_data_output = output[70:]

regr = linear_model.LinearRegression()

regr.fit(train_data_input, train_data_output)
print "\nppn as Numeric"
print("Root mean square error: %.2f"
      % np.sqrt(np.mean((regr.predict(test_data_input) - test_data_output) ** 2)))

test_data_output = np.array(test_data_output)
predicted_output = (regr.predict(test_data_input))
error = 0
for i in range(len(test_data_output)):
      error += np.abs(predicted_output[i] - test_data_output[i])

print "Mean absolute error",error / float(len(test_data_output))