import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.metrics import median_absolute_error

#Would memory always be in GB?

data = pd.read_csv('/Users/ramkishore31/PycharmProjects/workFlow/IPPD/data/ls002_02_25_16.csv')

output = data['SystemTime'] + data['UserTime']

#Remove the %symbol from CPU usage and convert it to integer.
data['PercentCPU'] = [int(i[:-1])for i in data['PercentCPU']]

#Remove G from memory and convert it to integer.
data['mem'] = [int(i[:-1])for i in data['mem']]

data_mem = data.loc[data['ppn']  == 8]
'''
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
'''


data = data.drop('SystemTime', 1)
data = data.drop('UserTime', 1)
data = data.drop('CommandTimed', 1)
data = data.drop('WallTime', 1)
data = data.drop('SampleName', 1)
data = data.drop('trial', 1)

data = np.array(data)

accuracy = []
absolute_error = []

kf = KFold(len(data), n_folds=5)
for train_index, test_index in kf:
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = output[train_index], output[test_index]

    accuracy.append(np.std(y_test))

kf = KFold(len(data), n_folds=5)
for train_index, test_index in kf:
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = output[train_index], output[test_index]

    svm_model = svm.SVR()
    svm_model.fit(X_train, y_train)
    accuracy.append(np.sqrt(np.mean((svm_model.predict(X_test) - y_test) ** 2)))
    absolute_error.append(median_absolute_error(svm_model.predict(X_test), y_test))

kf = KFold(len(data), n_folds=5)
for train_index, test_index in kf:
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = output[train_index], output[test_index]

    lasso = linear_model.Lasso(alpha = 0.01)
    lasso.fit(X_train,y_train)
    accuracy.append(np.sqrt(np.mean((lasso.predict(X_test) - y_test) ** 2)))
    absolute_error.append(median_absolute_error(lasso.predict(X_test), y_test))

kf = KFold(len(data), n_folds=5)
for train_index, test_index in kf:
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = output[train_index], output[test_index]

    ridge = linear_model.Ridge(alpha = 0.1)
    ridge.fit(X_train,y_train)
    accuracy.append(np.sqrt(np.mean((ridge.predict(X_test) - y_test) ** 2)))
    absolute_error.append(median_absolute_error(ridge.predict(X_test), y_test))

kf = KFold(len(data), n_folds=5)
for train_index, test_index in kf:
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = output[train_index], output[test_index]

    rf_model = RandomForestRegressor(random_state=0, n_estimators=20)
    rf_model.fit(X_train,y_train)
    accuracy.append(np.sqrt(np.mean((rf_model.predict(X_test) - y_test) ** 2)))
    absolute_error.append(median_absolute_error(rf_model.predict(X_test), y_test))

kf = KFold(len(data), n_folds=5)
for train_index, test_index in kf:
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = output[train_index], output[test_index]

    ab_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300)
    ab_model.fit(X_train,y_train)
    accuracy.append(np.sqrt(np.mean((ab_model.predict(X_test) - y_test) ** 2)))
    absolute_error.append(median_absolute_error(ab_model.predict(X_test), y_test))


print "Root mean square errors for different models...."
print "Naive model... ",np.asarray(accuracy[:4]).mean()
print "Linear Regression with L1 regularization(Lasso)... ",np.asarray(accuracy[5:9]).mean()
print "Linear Regression with L2 regularization(Ridge)... ",np.asarray(accuracy[10:14]).mean()
print "Support Vector Machine... ",np.asarray(accuracy[15:19]).mean()
print "Random Forest... ",np.asarray(accuracy[20:24]).mean()
print "Adaboost...",np.asarray(accuracy[25:29]).mean()

print "\n"
print "Mean absolute error for different models...."
print "Linear Regression with L1 regularization(Lasso)... ",np.asarray(absolute_error[:4]).mean()
print "Linear Regression with L2 regularization(Ridge)... ",np.asarray(absolute_error[5:9]).mean()
print "Support Vector Machine... ",np.asarray(absolute_error[10:14]).mean()
print "Random Forest... ",np.asarray(absolute_error[15:19]).mean()
print "Adaboost...",np.asarray(absolute_error[20:24]).mean()





