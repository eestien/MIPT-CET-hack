# Предсказыает режим работы (НЕФ/НАГ) в зависимости от значения "Закачка, м3", 'Попутный газ, м3', "Обводненность (вес), %"
from sklearn import svm
import pandas as pd
import config as cf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from joblib import dump, load
import numpy as np

def SVM():
    params_for_mode_prediction = ["Закачка, м3", 'Попутный газ, м3', "Обводненность (вес), %"]
    data = pd.read_excel(cf.base_dir+cf.train_raw)
    test_data = pd.read_excel(cf.base_dir+cf.test_raw)

    data = data[data['Время работы, ч'] != 0].fillna(0)
    data = data[data['Характер работы'] != 'НЕФ/НАГ']

    '''
    test_data = test_data[test_data['Время работы, ч'] != 0].fillna(0)
    test_data = test_data[test_data['Характер работы'] != 'НЕФ/НАГ']

    lb = LabelBinarizer()
    X_train = data[params_for_mode_prediction]
    y_train = data['Характер работы']
    y_train = lb.fit_transform(y_train)

    X_test_data = test_data[params_for_mode_prediction]


    X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X_train, y_train, test_size=0.3)
    print('SVM is fitting......')
    model_svc = svm.SVC(verbose=10, kernel='linear')
    model_svc.fit(X_train_sp, y_train_sp)

    pred = model_svc.predict(X_test_sp)
    print('Accuracy metric is testing')
    accu_percent = accuracy_score(y_test_sp, pred, model_svc) * 100
    print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))

    # Save model
    dump(model_svc, cf.base_dir + '/models/SVM.joblib')

    model = load(cf.base_dir + '/models/SVM.joblib')
    pred_real = model.predict(X_test_data)
    pred_real = lb.inverse_transform(pred_real)
    df = pd.DataFrame(pred_real, columns=['ouput'])
    print(df)
    '''


    ################################
    ##           MAXIMIZE         ##
    ################################

    # Регрессионная модель, предсказывающая количесвто чистой нефти при параметрах 'Жидкость, т', 'Вода, т'
    # TODO: Локализовать регрессионную модель по конкретным скважинам и "поиграться" с режимами этих насосов
    params_for_maximize = ['Обводненность (вес), %', 'Время работы, ч', 'Забойное давление', 'Давление на приеме', 'Пластовое давление']

    # for x in data[['Забойное давление']]:
      #  print(x)
    #data = data[data[['Забойное давление'] != 'datetime.datetime' in str(type(x))]]
    #data = data[params_for_maximize]
    X_train_max = data[params_for_maximize].fillna(0)#.applymap(lambda x: int(float(x)))
    #X_train_max["Способ эксплуатации"] = X_train_max["Способ эксплуатации"].astype("category")
    y_train_max = data['Нефть, т'].astype(int)

    X_train_max_sp, X_test_max_sp, y_train_max_sp, y_test_max_sp = train_test_split(X_train_max, y_train_max, test_size=0.3)
    '''
    # Set Classifier
    clf = LogisticRegression(C=0.01, penalty='l1')

    # Grid Search
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    logreg_cv = RandomizedSearchCV(clf, grid, cv=2, n_iter=3)
    logreg_cv.fit(X_train_max_sp, y_train_max_sp)

    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)
    '''
    '''
    # Fitting Logistic Regression to the Training set
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    '''
    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression(solver='lbfgs', max_iter=200)
    logisticRegr.fit(X_train_max_sp, y_train_max_sp)
    print('score', logisticRegr.predict(X_test_max_sp))
    print('score', logisticRegr.score(X_test_max_sp, y_test_max_sp))

    '''
    from sklearn.ensemble import RandomForestRegressor

    regr = RandomForestRegressor(max_depth=2, random_state=42, n_estimators=100)
    regr.fit(X_train_max_sp, y_train_max_sp)
    print(regr.feature_importances_)
    pred = regr.predict(X_test_max_sp)
    ac = r2_score(y_test_max_sp, pred, multioutput='variance_weighted')
    print('r2_score: ', ac)
    '''
SVM()