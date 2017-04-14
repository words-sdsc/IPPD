
# coding: utf-8

# 1   LassoCVModel (filename):
# 
# 2   OMPCVModel (filename):
# 
# 3   RidgeCVModel (filename):
# 
# 4   ElasticNetCVModel (filename):
# 
# 5   GradientBoostingCVModel (filename):
# 
# 6   RandomForestCVModel (filename):
# 
# 7   SVRSigmoidCVModel (filename):
# 
# 8   SVRRbfCVModel (filename):

# In[5]:

from sklearn.metrics import mean_squared_error 
import numpy as np
c = 0

def rmse_scorer(model, X, y):
    import sys
    import pandas as pd
    global c
    
    y_predict = model.predict(X)
    
    if(True in np.isnan(y_predict)):
        return sys.maxsize
    
    k = np.sqrt(mean_squared_error(y, y_predict))
    c = c+1
    
    return k


# <<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>>
# <<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>>
# <<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>>

# In[ ]:

#4
def RandomForestCVModel(filename, scale=False):
    #open file and get the dictionary
    import pickle
    from sklearn import preprocessing
    from sklearn.ensemble.forest import RandomForestRegressor
    from numpy.random import RandomState
    from sklearn.model_selection import GridSearchCV
    from IPython.display import display

    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################  
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    ##############################################################
    tuned_parameters = []
    tuned_parameters.append( { 
                              "n_estimators" :[2000, 4000, 6000, 10000]
                            })
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    grdsurch = GridSearchCV(RandomForestRegressor(n_estimators=6000, criterion='mse', 
                                                  max_depth=None, 
                                                  min_samples_split=2, 
                                                  min_samples_leaf=1, 
                                                  min_weight_fraction_leaf=0.0, 
                                                  max_features='auto', 
                                                  max_leaf_nodes=None, 
                                                  min_impurity_split=1e-07, 
                                                  bootstrap=True, oob_score=True, 
                                                  n_jobs=-1, random_state=None, 
                                                  verbose=0, warm_start=False), 
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
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was Random Forest


# In[ ]:

#3 OK
def GradientBoostingCVModel(filename, scale=False):
    #open file and get the dictionary
    import pickle
    from sklearn import preprocessing
    from sklearn.ensemble import GradientBoostingRegressor
    from numpy.random import RandomState
    from sklearn.model_selection import GridSearchCV
    from IPython.display import display


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################  
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))

    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    ##############################################################
    tuned_parameters = [     {  "n_estimators" :[2000, 4000, 6000, 10000],
                                "loss" : ['ls'],
                                "learning_rate": [0.001, 0.005, 0.01]
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

    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was GradientBoosting


# In[ ]:

def initialize_c():
    # c counts number of times scorer is called
    global c
    c=0


# In[ ]:

#5
def RidgeCVModel(filename, scale=True):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV 
    from IPython.display import display


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################  
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    
    ############################################################## RidgeL2
    tuned_parameters = []
    tuned_parameters.append( {'alpha' : np.logspace(-35, +25, 100) } ) 
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print(tuned_parameters)
    print("##########################")

    # Ridge Regression (L2)
    grdsurch = GridSearchCV(Ridge(alpha=1.0, fit_intercept=True, 
                             normalize=False, copy_X=True, max_iter=None, tol=1e-6, 
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
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    print("\n\n scorer is called: %d times \n\n" % c)
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was RidgeCVModel


# In[ ]:

#6
def ElasticNetCVModel(filename, scale=True):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from IPython.display import display


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################  
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    ############################################################## ElasticNet
    tuned_parameters = []
    tuned_parameters.append( { 
                               'alpha'   : np.logspace(-25, +25, 40), 
                               'l1_ratio': [0.9, 0.75, 0.5, .25, .1]  
                             }
                           ) 
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print(tuned_parameters)
    print("##########################")

    # Elastic Net (L1 + L2)
    # Linear regression with combined L1 and L2 priors as regularizer
    #
    grdsurch = GridSearchCV(ElasticNet(alpha=3.7926901907322537e-14, copy_X=True, fit_intercept=True,
      l1_ratio=0.9, max_iter=10000000.0, normalize=False, positive=False,
      precompute=False, random_state=None, selection='cyclic', tol=1e-20,
      warm_start=False), 
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
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was ElasticNet


# In[16]:

def get_svr_parameters():
    # parameters used by all SVR
    
    _parameters = []
    _parameters.append({
                             'C'      : np.logspace(-1,5,5),
                             'epsilon': np.array([ 50 ]),
                             'degree' : np.arange(3,20,15),
                             'coef0'  : np.array([ 0.0 ]),
                             'gamma'  : np.logspace(0, 15, 5)        
                            })
    return _parameters


# In[18]:

#7
def SVRPolyCVModel(filename, scale=True):
    #open file and get the dictionary
    import pickle
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    from IPython.display import display
    import pandas as pd


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
    

    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################  
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
        
    # -- a -- ‘poly’ 
    # 1111111111111111
    ##############################################################
    tuned_parameters = get_svr_parameters()
    

    
    ##############################################################
    
    print("# Tuning hyper-parameters... ")
    print()

    grdsurch = GridSearchCV(SVR(kernel='poly', degree=3, 
                                gamma='auto', coef0=0.0, tol=1e-2, C=1.0, 
                                epsilon=20, shrinking=False, cache_size=20*1024, 
                                verbose=False, max_iter=1e9), 
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
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was SVRPoly


# In[19]:

#8
def SVRSigmoidCVModel(filename, scale=False):
    #open file and get the dictionary
    import pickle
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    from IPython.display import display
    


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################   
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    # -- b -- ‘sigmoid’
    # 222222222222222222
    ##############################################################
    tuned_parameters = get_svr_parameters()
    
    '''
    tuned_parameters.append({
                             'gamma'  : np.logspace(-15, 3, 1),
                             'C'      : np.logspace(-5, 15, 1)
                            }
                            )    
    '''
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    grdsurch = GridSearchCV(SVR(kernel='sigmoid', degree=3, 
                                gamma='auto', coef0=0.0, tol=1e-7, C=1.0, 
                                epsilon=0.1, shrinking=False, cache_size=20*1024, 
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
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was SVRSigmoid


# In[20]:

#9

def SVRLinearCVModel(filename, scale=False):
    #open file and get the dictionary
    import pickle
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    from IPython.display import display


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################  
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    # -- c -- ‘linear’
    # 3333333333333333333
    ##############################################################
    tuned_parameters = get_svr_parameters()
    
    '''
    tuned_parameters.append({
                             # 'gamma'  : np.logspace(-15, 3, 5),
                             'C'      : np.logspace(-15, -5, 5)
                            })
    '''
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    grdsurch = GridSearchCV(SVR(kernel='linear', degree=3, 
                                gamma='auto', coef0=0.0, tol=1e-7, C=1.0, 
                                epsilon=0.1, shrinking=False, cache_size=20*1024, 
                                verbose=False, max_iter=-1), 
                       tuned_parameters, 
                       cv=3, 
                       n_jobs=-1, 
                       scoring=rmse_scorer)
    print('Starting grdsurch.fit(X_train, y_train)')
    
    grdsurch.fit(X_train, y_train)

    print("\n Best parameters set found on development set:")
    print()
    print(grdsurch.best_params_)
    
    print(grdsurch.best_estimator_)
    print()
    rmse_cv = grdsurch.best_score_

    #Reporting Score on Test Set
    model               = grdsurch.best_estimator_
    reporting_testscore = rmse_scorer(model, X_test, y_test)
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was SVRLinear


# In[21]:

#10

def SVRRbfCVModel(filename, scale=True):
    #open file and get the dictionary
    import pickle
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    from IPython.display import display


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################  
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    # -- d -- ‘rbf’
    # 44444444444444444444
    ##############################################################
    tuned_parameters = get_svr_parameters()
    
    '''
    tuned_parameters.append({ 
                             'gamma'  : np.logspace(-15, 3, 5),
                             'C'      : np.logspace(-5, 15, 5)
                             })
    '''
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()

    grdsurch = GridSearchCV(SVR(kernel='rbf', degree=3, 
                                gamma='auto', coef0=0.0, tol=1e-7, C=1.0, 
                                epsilon=0.1, shrinking=False, cache_size=100*1024, 
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
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was SVRrbf


# In[ ]:

## OK below this


# In[ ]:

def scale_this(scale, X_trai0, X_tes0):
    ############ scaling of features #################
    
    from sklearn import preprocessing
    import copy
    
    X_trai = X_trai0.copy()
    X_tes  = X_tes0.copy()
    
    if(scale):
        ##################################################
        # Scale X
        print('\n{{ scale_this: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Scaling X_train and X_test... }}\n')
        standard_scaler = preprocessing.StandardScaler()
        X_train = standard_scaler.fit_transform(X_trai)
        X_test  = standard_scaler.transform(X_tes)
        ##################################################
    else:
        X_train = X_trai
        X_test  = X_tes
        ##################################################
    return X_train, X_test


# In[ ]:

import numpy as np
np.logspace(2, 3, 30)


# In[ ]:

#1 OK
def LassoCVModel(filename, scale=True):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
    from IPython.display import display


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################   
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))
    
    
    ############################################################## Lasso
    tuned_parameters = []
    tuned_parameters.append( {'alpha' : np.logspace(-4, 10, 50),
                              'precompute' : [True, False]
                             } )
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print(tuned_parameters)
    print("##########################")

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
    
    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was Lasso (L1)


# In[ ]:

#2 OK
def OMPCVModel(filename, scale=False):
    #open file and get the dictionary
    import pickle
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    from IPython.display import display


    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    #extract X_train, y_train, X_test, t_test
    X_trai  = data['X_train']
    y_train = data['y_train']
    X_tes   = data['X_test']
    y_test  = data['y_test']
        
    ############ scaling of features #################
    
    X_train, X_test = scale_this(scale, X_trai, X_tes)
    
    ##################################################   
    print("Dataset size read: train %d and test %d \n" %(len(y_train), len(y_test)))

    #Normalize
    #X_train = preprocessing.normalize(X_train, norm='l1')
    #X_test  = preprocessing.normalize(X_test,  norm='l1')
    
    #
    ##############################################################
    tuned_parameters = []
    tuned_parameters.append({'tol' : [1e-4],
                             'n_nonzero_coefs' : [14, 28]
                            
                            })
    
    ##############################################################
    
    print("# Tuning hyper-parameters ")
    print()
    
    # OMP
    grdsurch = GridSearchCV(OrthogonalMatchingPursuit(n_nonzero_coefs=3, 
                                                      tol=1e-15, fit_intercept=True, 
                                                      normalize=False, precompute='auto'), 
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

    ###########################
    # added for measure predictions on X_test_A, X_test_B ...
    print('Full Test Set: %d' % len(y_test))
    display(data['y_test'])
    display(model.predict(X_test))
    
    reporting_testscoreA = None
    reporting_testscoreB = None
    reporting_testscoreC = None
    reporting_testscoreD = None
    test_mean_y_comparingA = None
    test_mean_y_comparingB = None
    test_mean_y_comparingC = None
    test_mean_y_comparingD = None        
        
    if('y_test_A' in data):
        print('A: %d' % len(data['y_test_A']))
        X_train, X_test_A = scale_this(scale, X_trai, data['X_test_A'])
        
        reporting_testscoreA = rmse_scorer(model, X_test_A, data['y_test_A'])
        display(data['y_test_A'])
        display(model.predict(X_test_A))
        test_mean_y_comparingA = data['y_test_A'].mean()

    if('y_test_B' in data):
        print('B: %d' % len(data['y_test_B']))
        X_train, X_test_B = scale_this(scale, X_trai, data['X_test_B'])
        
        reporting_testscoreB = rmse_scorer(model, X_test_B, data['y_test_B'])
        display(data['y_test_B'])
        display(model.predict(X_test_B))
        test_mean_y_comparingB = data['y_test_B'].mean()

    if('y_test_C' in data):
        print('C: %d' % len(data['y_test_C']))
        X_train, X_test_C = scale_this(scale, X_trai, data['X_test_C'])
        
        reporting_testscoreC = rmse_scorer(model, X_test_C, data['y_test_C'])
        display(data['y_test_C'])
        display(model.predict(X_test_C))
        test_mean_y_comparingC = data['y_test_C'].mean()

    if('y_test_D' in data):
        print('D: %d' % len(data['y_test_D']))
        X_train, X_test_D = scale_this(scale, X_trai, data['X_test_D'])
        
        reporting_testscoreD = rmse_scorer(model, X_test_D, data['y_test_D'])
        display(data['y_test_D'])
        display(model.predict(X_test_D))
        test_mean_y_comparingD = data['y_test_D'].mean()
    
    return {filename: {'train_rmse_cv_picking': rmse_cv, 
                       'test_rmse_reporting' : reporting_testscore,
                       'test_rmse_reportingA': reporting_testscoreA,
                       'test_rmse_reportingB': reporting_testscoreB,
                       'test_rmse_reportingC': reporting_testscoreC,
                       'test_rmse_reportingD': reporting_testscoreD,
                       'test_mean_y_comparing': y_test.mean(),
                       'test_mean_y_comparingA': test_mean_y_comparingA,
                       'test_mean_y_comparingB': test_mean_y_comparingB,
                       'test_mean_y_comparingC': test_mean_y_comparingC,
                       'test_mean_y_comparingD': test_mean_y_comparingD,
                       'model': model
                      }}

#this was OMPCV


# In[ ]:

print()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



