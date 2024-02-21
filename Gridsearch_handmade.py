import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def maden_CV(X_train, y_train, name, model, TRAINSAMPLER, TESTSAMPLER, seed) :
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  ros = RandomOverSampler(random_state=seed)
  i = 1
  stdlist = []
  if TRAINSAMPLER * TESTSAMPLER == 1 :
    X_train, y_train = ros.fit_resample(X_train, y_train)

  for train_index, test_index in skf.split(X_train, y_train):
      X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
      y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

      if TRAINSAMPLER + TESTSAMPLER == 1 :
        X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

      # 모델 학습 및 예측
      model.fit(X_train_fold, y_train_fold)
      predictions = model.predict(X_test_fold)

      # 결과 출력
      f1 = f1_score(y_test_fold, predictions)
      stdlist.append(f1)
      i += 1
  means = np.mean(stdlist)
  stds = np.std(stdlist)
  
  return [means, stds]

def maden_Grid(X_train, y_train, MODELNAME, TRAINSAMPLER, TESTSAMPLER, seed) :

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  ros = RandomOverSampler(random_state=seed)

  if TRAINSAMPLER == 1 :
      trainsampling = 'OVERSAMPING'
  else :
      trainsampling = 'ORIGINAL'

  if TESTSAMPLER == 1 :
      testsampling = 'OVERSAMPING'
  else :
      testsampling = 'ORIGINAL'

  print(MODELNAME, trainsampling, testsampling)

###################################################################################################################################################

  if MODELNAME == 'KNN' :
    param ={
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'leaf_size': [20, 30, 50]
    }

    result_temp = []

    for i in tqdm(param['n_neighbors']) :
        for j in param['weights'] :
            for h in param['p'] :
                for k in param['leaf_size'] :
                    model = KNeighborsClassifier(n_neighbors = i, weights = j, p = h, leaf_size = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'XGB' :
    param = {
      'n_estimators': [50, 100, 200],
      'learning_rate': [0.01, 0.1, 0.3],
      'max_depth': [3, 5, 7],
      'min_child_weight': [1, 3, 5]
      }

    result_temp = []

    for i in tqdm(param['n_estimators']) :
        for j in param['learning_rate'] :
            for h in param['max_depth'] :
                for k in param['min_child_weight'] :
                    model = XGBClassifier(n_estimators = i, learning_rate = j, max_depth = h, min_child_weight = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'RF' :
    param = {
        'n_estimators' : [10, 100],
        'max_depth' : [6, 8],
        'min_samples_leaf' : [8, 12],
        'min_samples_split' : [8, 16]
    }

    result_temp = []

    for i in tqdm(param['n_estimators']) :
        for j in param['max_depth'] :
            for h in param['min_samples_leaf'] :
                for k in param['min_samples_split'] :
                    model = RandomForestClassifier(n_estimators = i, max_depth = j, min_samples_leaf = h, min_samples_split = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'ET' :
    param = {
        'n_estimators' : [10, 15, 20],
        'max_depth' : [2, 3, 4],
        'min_samples_leaf' : [2, 3, 4, 5],
        'min_samples_split' : [2, 3, 4, 5]
    }

    result_temp = []

    for i in tqdm(param['n_estimators']) :
        for j in param['max_depth'] :
            for h in param['min_samples_leaf'] :
                for k in param['min_samples_split'] :
                    model = ExtraTreesClassifier(n_estimators = i, max_depth = j, min_samples_leaf = h, min_samples_split = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'NN' :
    param = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'learning_rate': ['constant', 'adaptive', 'invscaling']
        }

    result_temp = []

    for i in tqdm(param['hidden_layer_sizes']) :
        for j in param['activation'] :
            for h in param['solver'] :
                for k in param['learning_rate'] :
                    model = MLPClassifier(hidden_layer_sizes= i, activation = j, solver = h, learning_rate = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])


##################################################################################################################################################

  elif MODELNAME == 'LDA' :
    param = {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5],
        'n_components': [1, 2, 3],
        'priors': [None, [0.3, 0.7], [0.4, 0.6]]
        }

    result_temp = []

    for i in tqdm(param['solver']) :
        for j in param['shrinkage'] :
            for h in param['n_components'] :
                for k in param['priors'] :
                    model = LinearDiscriminantAnalysis(solver = i, shrinkage = j, n_components = h, priors = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'LGBM' :
    param = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_child_samples': [10, 20, 30]
        }

    result_temp = []

    for i in tqdm(param['n_estimators']) :
        for j in param['learning_rate'] :
            for h in param['max_depth'] :
                for k in param['min_child_samples'] :
                    model = LGBMClassifier(n_estimators = i, learning_rate = j, max_depth = h, min_child_samples = k, verbose = -1)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'LR' :
    param = {
        'penalty': ['l1', 'l2', 'none'],
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs', 'newton-cg'],
        'max_iter': [100, 200, 300]
        }

    result_temp = []

    for i in tqdm(param['penalty']) :
        for j in param['C'] :
            for h in param['solver'] :
                for k in param['max_iter'] :
                    model = LogisticRegression(penalty = i, C = j, solver = h, max_iter = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'CART' :
    param = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
        }

    result_temp = []

    for i in tqdm(param['criterion']):
        for j in param['splitter']:
            for h in param['max_depth']:
                for k in param['min_samples_split']:
                    model = DecisionTreeClassifier(criterion=i, splitter=j, max_depth=h, min_samples_split=k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################

  elif MODELNAME == 'SVM' :
    param = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'degree': [2, 3, 4]
        }

    result_temp = []

    for i in tqdm(param['C']) :
        for j in param['kernel'] :
            for h in param['gamma'] :
                for k in param['degree'] :
                    model = SVC(C = i, kernel = j, gamma = h, degree = k)

                    stdlist = []

                    if TRAINSAMPLER * TESTSAMPLER == 1 :
                      X_train, y_train = ros.fit_resample(X_train, y_train)

                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                        y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                        if TRAINSAMPLER + TESTSAMPLER == 1 :
                          X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)

                        model.fit(X_train_fold, y_train_fold)
                        predictions = model.predict(X_test_fold)
                        f1 = f1_score(y_test_fold, predictions)
                        stdlist.append(f1)
                    result_temp.append([MODELNAME, trainsampling, testsampling, i, j, h, k, np.mean(stdlist), np.std(stdlist)])

    Result_grid = pd.DataFrame(result_temp, columns=['ModelName', 'trainsampling', 'testsampling', 'param1', 'param2', 'param3', 'param4', 'mean', 'std'])

##################################################################################################################################################


  Result_grid = Result_grid.sort_values('mean', ascending=False).reset_index(drop=True)



  return Result_grid

def maden_finalmodel (modelname, X_train, y_train, X_test, y_test, SAMPLER, param1, param2, param3, param4) :
    if modelname == 'KNN' :
        finalmodel = KNeighborsClassifier(n_neighbors = param1, weights = param2, p = param3, leaf_size = param4)
    elif modelname == 'XGB' :
        finalmodel = XGBClassifier(n_estimators = param1, learning_rate = param2, max_depth = param3, min_child_weight = param4)
    elif modelname == 'RF' :
        finalmodel = RandomForestClassifier(n_estimators = param1, max_depth = param2, min_samples_leaf = param3, min_samples_split = param4)
    elif modelname == 'ET' :
        finalmodel = ExtraTreesClassifier(n_estimators = param1, max_depth = param2, min_samples_leaf = param3, min_samples_split = param4)
    elif modelname == 'NN' :
        finalmodel = MLPClassifier(hidden_layer_sizes= param1, activation = param2, solver = param3, learning_rate = param4)
    elif modelname == 'LDA' :
        finalmodel = LinearDiscriminantAnalysis(solver = param1, shrinkage = param2, n_components = param3, priors = param4)
    elif modelname == 'LGBM' :
        finalmodel = LGBMClassifier(n_estimators = param1, learning_rate = param2, max_depth = param3, min_child_samples = param4, verbose = -1)
    elif modelname == 'LR' :
        finalmodel = LogisticRegression(penalty = param1, C = param2, solver = param3, max_iter = param4)
    elif modelname == 'CART' :
        finalmodel = DecisionTreeClassifier(criterion=param1, splitter=param2, max_depth=param3, min_samples_split=param4)
    elif modelname == 'SVM' :
        finalmodel = SVC(C = param1, kernel = param2, gamma = param3, degree = param4)
    
    if SAMPLER != 0 :
        rs = RandomOverSampler(random_state=42)
        X_train, y_train = rs.fit_resample(X_train, y_train)


    # 모델 학습 및 예측
    finalmodel.fit(X_train, y_train)
    predictions = finalmodel.predict(X_test)
    result = f1_score(y_test, predictions)
    return result