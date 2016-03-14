import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

#Would memory always be in GB?

data = pd.read_csv('/Users/ramkishore31/PycharmProjects/workFlow/IPPD/data/ls002_02_25_16.csv')

output = data['SystemTime'] + data['UserTime']

#Remove the %symbol from CPU usage and convert it to integer.
data['PercentCPU'] = [int(i[:-1])for i in data['PercentCPU']]

#Remove G from memory and convert it to integer.
data['mem'] = [int(i[:-1])for i in data['mem']]

data_mem = data.loc[data['ppn']  == 8]
fig = plt.figure()
plt.scatter(data_mem['mem'],data_mem['UserTime'] + data_mem['SystemTime'])
fig.suptitle('Variation of time with memory for 8 core machines', fontsize=20)
plt.xlabel('Memory in G', fontsize=18)
plt.ylabel('Total time', fontsize=16)
fig.savefig('mem_time.jpg')
plt.show()



data_mem = data.loc[data['mem']  == 110]
fig = plt.figure()
plt.scatter(data_mem['ppn'],data_mem['UserTime'] + data_mem['SystemTime'])
fig.suptitle('Variation of time with number of cores for machines with 110G memory', fontsize=20)
plt.xlabel('Number of cores', fontsize=18)
plt.ylabel('Total time', fontsize=16)
fig.savefig('cpu_time.jpg')
plt.show()



data = data.drop('SystemTime', 1)
data = data.drop('UserTime', 1)
data = data.drop('CommandTimed', 1)
data = data.drop('WallTime', 1)
data = data.drop('SampleName', 1)
data = data.drop('trial', 1)

data = np.array(data)

kf = KFold(len(data), n_folds=5)
for train_index, test_index in kf:
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = output[train_index], output[test_index]


regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

print("Root mean square error: %.2f"
      % np.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))

print np.std(y_test)






