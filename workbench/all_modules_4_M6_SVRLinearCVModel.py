
# coding: utf-8

# In[23]:

# SVRLinearCVModel('datasetIPPD.pickle')


# Dataset size read: train 256 and test 65 
# 
# # Tuning hyper-parameters 
# 
# Starting grdsurch.fit(X_train, y_train)
# 
# Best parameters set found on development set:
# 
# {'C': 1.0000000000000001e-05, 'epsilon': 100000.0, 'gamma': 1.0000000000000001e-05, 'kernel': 'linear'}
# SVR(C=1.0000000000000001e-05, cache_size=1024, coef0=0.0, degree=3,
#   epsilon=100000.0, gamma=1.0000000000000001e-05, kernel='linear',
#   max_iter=-1, shrinking=False, tol=1e-07, verbose=False)
# 
# CPU times: user 927 ms, sys: 164 ms, total: 1.09 s
# Wall time: 1.87 s
# 
# {'datasetIPPD.pickle': {'model': SVR(C=1.0000000000000001e-05, cache_size=1024, coef0=0.0, degree=3,
#     epsilon=100000.0, gamma=1.0000000000000001e-05, kernel='linear',
#     max_iter=-1, shrinking=False, tol=1e-07, verbose=False),
#   'test_mean_y_4_comparing': 1532.3461538461536,
#   'test_rmse_4_reporting': 1059.8422915291899,
#   'train_rmse_cv_4_picking': 1080.4666523912699}}
# 
# 

# # M6. Epsilon-Support Vector Regression

# In[24]:

from sklearn.metrics import mean_squared_error 
import numpy as np
def rmse_scorer(model, X, y): 
    y_predict = model.predict(X)
    k = np.sqrt(mean_squared_error(y, y_predict))
    return k


# In[25]:

def SVRLinearCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV

    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    #Normalize
    X_train = preprocessing.normalize(X_train, norm='l1')
    X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
    ##############################################################
    tuned_parameters = []
    
    
    tuned_parameters.append({'kernel' : ['linear'], 
                             'gamma'  : np.logspace(-5, 5, 5),
                             'C'      : np.logspace(-5, 5, 5),
                             'epsilon': np.logspace(-5, 5, 5) 
                            })

    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    grdsurch = GridSearchCV(SVR(kernel='rbf', degree=3, 
                                gamma='auto', coef0=0.0, tol=1e-7, C=1.0, 
                                epsilon=0.1, shrinking=False, cache_size=1024, 
                                verbose=False, max_iter=-1), 
                       tuned_parameters, 
                       cv=3, 
                       n_jobs=-1, 
                       scoring=rmse_scorer)
    print('Starting grdsurch.fit(X_train, y_train)')
    
    grdsurch.fit(X_train, y_train)

    print("\nBest parameters set found on development set:")
    print()
    print(grdsurch.best_params_)
    
    print(grdsurch.best_estimator_)
    print()
    rmse_cv = grdsurch.best_score_

    #Reporting Score on Test Set
    model               = grdsurch.best_estimator_
    reporting_testscore = rmse_scorer(model, X_test, y_test)
    
    return {filename: {'train_rmse_cv_4_picking': rmse_cv, 
                       'test_rmse_4_reporting': reporting_testscore, 
                       'test_mean_y_4_comparing': y_test.mean(),
                       'model': model
                      }}


# In[ ]:




# In[ ]:



