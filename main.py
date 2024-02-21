# 패키지
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold

from Gridsearch_handmade import maden_CV, maden_Grid, maden_finalmodel


val_data = pd.read_csv('C:/Users/daeun/Desktop/BDA/Dacon/사기탐지/data/val.csv')
  
# 데이터 분할
X = val_data.drop(['ID', 'Class'], axis=1)
y = val_data['Class']

# 선택할 스케일러 지정 (1: StandardScaler, 2: MinMaxScaler, 3: RobustScaler, 4: MaxAbsScaler)
selected_scaler_number = 1

# 상관관계의 절댓값이 높은 상위 n개의 칼럼 선택 (예: n=5)
COLSEL = 7

# 선택한 스케일러로 데이터 스케일링
if selected_scaler_number == 1:
  selected_scaler = StandardScaler()
  scaler = 'STANDARDSCALER'
  X_scaled = selected_scaler.fit_transform(X)
elif selected_scaler_number == 2:
  selected_scaler = MinMaxScaler()
  scaler = 'MINMAXSCALER'
  X_scaled = selected_scaler.fit_transform(X)
elif selected_scaler_number == 3:
  selected_scaler = RobustScaler()
  scaler = 'ROBUSTSCALER'
  X_scaled = selected_scaler.fit_transform(X)
elif selected_scaler_number == 4:
  selected_scaler = MaxAbsScaler()
  scaler = 'MAXABSSCALER'
  X_scaled = selected_scaler.fit_transform(X)
else:
  scaler = 'NONE'
  X_scaled = X

# 스케일링된 데이터를 데이터프레임으로 변환 (옵션)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 상관관계 계산
correlations = X.corrwith(y)

top_n_correlations = correlations.abs().nlargest(COLSEL).index

# 선택된 칼럼으로 데이터프레임 생성
selected_columns_df = X_scaled_df[top_n_correlations]

X = selected_columns_df

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

cvfold_train_index = []
cvfold_test_index = []

for X_index, y_index in skf.split(X, y):
  cvfold_train_index.append(X_index)
  cvfold_test_index.append(y_index)


# TEST data가 될 fold의 번호를 넣으시오. (0 ~ 4)
TESTNUM = 4

X_train = X.iloc[cvfold_train_index[TESTNUM]]
y_train = y.iloc[cvfold_train_index[TESTNUM]]

X_test = X.iloc[cvfold_test_index[TESTNUM]]
y_test = y.iloc[cvfold_test_index[TESTNUM]]

# spot check the algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('KNN',  KNeighborsClassifier()))
#Neural Network
models.append(('NN', MLPClassifier()))
# #Ensable Models
models.append(('LGBM', LGBMClassifier(verbose = -1)))
models.append(('XGB', XGBClassifier()))
# Bagging methods
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))

seed = 42

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

print('Cross Validating . . .')
Result_model = []
for train_sam, test_sam in [(0,0), (1,0), (1,1)] :
  if train_sam + test_sam == 0 :
    print('Original -> Original')
  elif train_sam + test_sam == 1 :
    print('Over -> Original')
  elif train_sam + test_sam == 2 :
    print('Over -> Over')
  mean = []
  std = []
  for name, model in tqdm(models):
    mean_temp, std_temp = maden_CV(X_train, y_train, name, model, train_sam, test_sam, seed)
    mean.append(mean_temp)
    std.append(std_temp)
  Result_model.append(pd.DataFrame({
      'ModelName': [name for name, _ in models],
      'Mean': mean,
      'Std': std
  }))
  
modelnames = []
for i in range(len(Result_model)) :
    Result_model[i] = Result_model[i].sort_values('Mean', ascending=False).reset_index(drop=True)
    modelnames.append(list(Result_model[i]['ModelName'][0:2]))

print('Hyper Parameter Tuning . . .')
Result_tuned = []
for train_sam, test_sam in [(0,0), (1,0), (1,1)] :
  Result_tuned.append(maden_Grid(X_train, y_train, modelnames[train_sam+test_sam][0], train_sam, test_sam, seed))
  Result_tuned.append(maden_Grid(X_train, y_train, modelnames[train_sam+test_sam][1], train_sam, test_sam, seed))
  
FINALMODELS = []
for i in range(len(Result_tuned)) :
    FINALMODELS.append(Result_tuned[i].iloc[0])
FINALMODELDF = pd.concat(FINALMODELS, ignore_index=True)


print('Test Predicting . . .')
finalresult = []
for finalmodel in tqdm(FINALMODELS) :
  finalresult.append(maden_finalmodel(finalmodel['ModelName'], X_train, y_train, X_test, y_test, 0, finalmodel['param1'], finalmodel['param2'], finalmodel['param3'], finalmodel['param4']))
  finalresult.append(maden_finalmodel(finalmodel['ModelName'], X_train, y_train, X_test, y_test, 1, finalmodel['param1'], finalmodel['param2'], finalmodel['param3'], finalmodel['param4']))


FINAL = pd.DataFrame(columns = ['ModelName','transam','testsam','param1','param2',
                        'param3','param4','mean','std','Datatrain','test'])
for i in range(12):
  j= int(i/2)
  if i%2 == 1 : 
    new_data =  {'ModelName': FINALMODELS[j]['ModelName'] , 'transam':FINALMODELS[j]['trainsampling'] ,'testsam':FINALMODELS[j]['testsampling'],
              'param1' :FINALMODELS[j]['param1'] ,'param2': FINALMODELS[j]['param2'],
                        'param3' : FINALMODELS[j]['param3'],'param4' : FINALMODELS[j]['param4'],
                        'mean' : FINALMODELS[j]['mean'],'std' : FINALMODELS[j]['std'],
                        'Datatrain' : 'ORIGINAL',
                                            'test' : finalresult[i]}
  elif i%2 == 0 :
    new_data =  {'ModelName': FINALMODELS[j]['ModelName'] , 'transam':FINALMODELS[j]['trainsampling'] ,'testsam':FINALMODELS[j]['testsampling'],
              'param1' :FINALMODELS[j]['param1'] ,'param2': FINALMODELS[j]['param2'],
                        'param3' : FINALMODELS[j]['param3'],'param4' : FINALMODELS[j]['param4'],
                        'mean' : FINALMODELS[j]['mean'],'std' : FINALMODELS[j]['std'],
                        'Datatrain' : 'Over',
                                            'test' : finalresult[i]}
  new_data = pd.DataFrame(new_data, index = [0])
  FINAL = pd.concat([FINAL, new_data])
  FINAL = FINAL.reset_index(drop= True)
  

FINAL = FINAL.sort_values('test', ascending = False)
FINAL = FINAL.reset_index(drop = True)
print(FINAL)