
# coding: utf-8

# In[1]:

from sklearn.metrics import mean_squared_error 
import numpy as np

def rmse_scorer(model, X, y): 
    y_predict = model.predict(X)
    k = np.sqrt(mean_squared_error(y, y_predict))
    return k


# 
# 1  def LassoCVModel(filename)
# 
# 2  def OMPCVModel(filename)
# 
# 3  def GradientBoostingCVModel(filename)
# 
# 4  def RandomForestCVModel(filename)
# 
# 5  def RidgeCVModel(filename)
# 
# 6  def ElasticNetCVModel(filename)
# 
# 7  def SVRPolyCVModel(filename)
# 
# 8  def SVRSigmoidCVModel(filename)
# 
# 9  def SVRLinearCVModel(filename)
# 
# 10 def SVRRbfCVModel(filename)
# 

# <<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>>
# <<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>>
# <<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>>

# In[54]:

#4
def RandomForestCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn import preprocessing
    from sklearn.ensemble.forest import RandomForestRegressor
    from numpy.random import RandomState
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
    
    ##############################################################
    tuned_parameters = []
    tuned_parameters.append( { 
                              "n_estimators" :[2000, 4000, 6000]
                            })
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    grdsurch = GridSearchCV(RandomForestRegressor(n_estimators=6000, criterion='mse', max_depth=None, 
                                                  min_samples_split=2, 
                                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                                  max_leaf_nodes=None, min_impurity_split=1e-07, 
                                                  bootstrap=True, oob_score=True, 
                                                  n_jobs=-1, random_state=None, verbose=0, warm_start=False), 
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


# In[72]:

#5
def RidgeCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import Ridge
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
    
    ##############################################################
    tuned_parameters = []
    tuned_parameters.append( {'alpha' : np.logspace(-15, -10, 100) })
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    # Ridge Regression (L2)
    grdsurch = GridSearchCV(Ridge(alpha=1.0, fit_intercept=True, 
                             normalize=False, copy_X=True, max_iter=None, tol=1e-20, 
                             solver='auto', random_state=None), 
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


# In[82]:

#6
def ElasticNetCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import ElasticNet
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
    
    ##############################################################
    tuned_parameters = []
    tuned_parameters.append( { 'alpha'   : np.logspace(-5, 1, 10), 
                               'l1_ratio': [.1, .5, .99]  
                             }
                           ) 
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    # Elastic Net (L1 + L2)
    # Linear regression with combined L1 and L2 priors as regularizer
    grdsurch = GridSearchCV(ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, 
                                       precompute=False, max_iter=1e7, copy_X=True, tol=1e-20, 
                                       warm_start=False, positive=False, 
                                       random_state=None, selection='cyclic'), 
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


# In[9]:

#7
def SVRPolyCVModel(filename):
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
    
    tuned_parameters.append({'kernel' : ['poly'], 
                             'gamma'  : np.logspace(-15, -5, 4),
                             'C'      : np.logspace(-15, -5, 4),
                             'epsilon': np.logspace(-15, -5, 4),
                             'degree' : [3,6,12,24,36]
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


# In[10]:

#8
def SVRSigmoidCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    import numpy as np

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
    
    tuned_parameters.append({'kernel' : ['sigmoid'], 
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


# In[11]:

#9

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


# In[12]:

#10


def SVRRbfCVModel(filename):
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

    tuned_parameters.append({'kernel' : ['rbf'], 
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


# In[53]:

## OK below this


# In[ ]:




# In[50]:

#1 OK
def LassoCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import Lasso
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
    
    ##############################################################
    tuned_parameters = []
    tuned_parameters.append( {'alpha' : np.logspace(-4, 10, 30),
                              'precompute' : [True, False]
                             } )
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    # Lasso (L1)
    grdsurch = GridSearchCV(Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, 
                                  copy_X=True, max_iter=1e7, tol=1e-6, warm_start=False, 
                                  positive=False, random_state=None, selection='random'), 
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
                      }
           }


# In[52]:

#2 OK
def OMPCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import OrthogonalMatchingPursuit
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
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
    ##############################################################
    tuned_parameters = []
    tuned_parameters.append({'tol' : [1e-20, 1e-15, 1e-11],
                             'n_nonzero_coefs' : [3, 7, 14, 28]
                            
                            })
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()
    
    # OMP
    grdsurch = GridSearchCV(OrthogonalMatchingPursuit(n_nonzero_coefs=None, 
                                                      tol=None, fit_intercept=True, 
                                                      normalize=True, precompute='auto'), 
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


# In[49]:

#3 OK
def GradientBoostingCVModel(filename):
    #open file and get the dictionary
    import pickle
    from sklearn import preprocessing
    from sklearn.ensemble import GradientBoostingRegressor
    from numpy.random import RandomState
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
    
    ##############################################################
    tuned_parameters = [     {  "n_estimators" :[6000],
                               "loss" : ['ls'],
                                "learning_rate": [0.005, 0.01, 0.001]
                             }
                       ]

    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    #Boosting
    grdsurch = GridSearchCV(GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=6000, subsample=1.0, 
                                                      criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
                                                      min_weight_fraction_leaf=0.0, max_depth=None, 
                                                      min_impurity_split=1e-07, 
                                                      init=None, random_state=None, max_features=None, alpha=0.9, 
                                                      verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'), 
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

