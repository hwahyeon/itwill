'''
 문) digits 데이터 셋을 이용하여 다음과 단계로 Pipeline 모델을 생성하시오.
  <단계1> dataset load
  <단계2> Pipeline model 생성
          - scaling : StndardScaler 클래스, modeing : SVC 클래스    
  <단계3> GridSearch model 생성
          - params : 10e-2~10e+2, 평가방법 : accuracy, 교차검정 : 5겹
          - CPU 코어 수 : 2개 
  <단계4> best score, best params 출력 
'''

from sklearn.datasets import load_digits # dataset 
from sklearn.svm import SVC # model
from sklearn.model_selection import GridSearchCV # gride search model
from sklearn.pipeline import Pipeline # pipeline
from sklearn.preprocessing import StandardScaler # dataset scaling

# 1. dataset load
digits = load_digits()
X, y = digits.data, digits.target
# data 확인 
X.max() # 16
X.min() # 0


# 2. pipeline model : step1(data 표준화) -> step2(model) 
pipe_model = Pipeline([('scal', StandardScaler()), 
                             ('svc', SVC(random_state=1))])


# parameter rage : 10e-2 ~ 10e+2
params = [ 0.01, 0.1, 1.0, 10.0, 100.0, 100.0]
# 선형/비선형 params
param_grid = [
    {'svc__C': params, 'svc__kernel': ['linear']}, 
    {'svc__C': params, 'svc__gamma': params, 'svc__kernel': ['rbf']}]

# 3. gride search model : pipeline model, 평가방법 CV=5
gs = GridSearchCV(estimator=pipe_model, param_grid = param_grid,
                  scoring='accuracy', cv=5, n_jobs=2) # n_jobs = CPU 코어 수
gs_model = gs.fit(X, y)

# 교차검정 결과 
gs_model.cv_results_["mean_test_score"]

# 4. best score, best params
print('best score=', gs_model.best_score_)
print('best params=', gs_model.best_params_)
'''
best score= 0.956032188177035
best params= {'svc__C': 10.0, 'svc__gamma': 0.01, 'svc__kernel': 'rbf'}
'''




