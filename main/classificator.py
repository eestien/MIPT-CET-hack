# Предсказыает режим работы (НЕФ/НАГ) в зависимости от значения "Закачка, м3", 'Попутный газ, м3', "Обводненность (вес), %"
from sklearn import svm
import pandas as pd
from Hack import config as cf
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import openpyxl

def SVM(datafile: str):
    data = 0
    test_data = 0
    if datafile.find('.xlsx') >= 0:
        data = pd.read_excel(cf.base_dir + cf.train_raw)
        test_data = pd.read_excel(cf.base_dir + cf.test_raw)

    if datafile.find('.csv') >= 0:
        data = pd.read_csv(cf.base_dir + cf.train_raw)
        test_data = pd.read_csv(cf.base_dir + cf.test_raw)
    params_for_mode_prediction = ["Закачка, м3", "Обводненность (вес), %"]

    data = data[data['Время работы, ч'] != 0].fillna(0)
    data = data[data['Характер работы'] != 'НЕФ/НАГ']


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
    return pred_real



    ################################
    ##           MAXIMIZE         ##
    ################################

    # Регрессионная модель, предсказывающая количесвто чистой нефти при параметрах 'Жидкость, т', 'Вода, т'
    # TODO: Локализовать регрессионную модель по конкретным скважинам и "поиграться" с режимами этих насосов

def clean(x):
    x = str(x)
    if x.find('0:00:00') >= 0:
        return np.nan
    else:
        return float(x)
    
def Regression(datafile_tr:str, datafile_ts:str, num_left = cf.num_left):
    params_for_maximize = ['Обводненность (), %', 'Время работы, ч', 'Забойное давление', 'Давление на приеме', 'Пластовое давление']
    data = 0
    test_data = 0
    # Check data format
    dt = str(datafile_tr)
    ts = str(datafile_ts)
    if dt.find('.xlsx') >= 0 or ts.find('.xlsx') >= 0:
        data = pd.read_excel(datafile_tr)
        test_data = pd.read_excel(datafile_ts)

    if dt.find('.csv') >= 0 or ts.find('.csv') >= 0:
        data = pd.read_csv(datafile_tr)
        test_data = pd.read_csv(datafile_ts)


    params_for_maximize = ['Обводненность (вес), %', 'Время работы, ч', 'Забойное давление', 'Давление на приеме',
                          'Пластовое давление', 'Нефть, т']
    X_max = data[params_for_maximize]
    X_max = X_max.fillna(X_max.median())

    X_max['Забойное давление'] = X_max['Забойное давление'].apply(clean)
    X_max['Давление на приеме'] = X_max['Давление на приеме'].apply(clean)

    X_max = X_max.fillna(X_max.median())
    y_max = X_max.pop('Нефть, т')

    X_train_max_sp, X_test_max_sp, y_train_max_sp, y_test_max_sp = train_test_split(X_max, y_max,
                                                                                    test_size=0.3)

    regr = RandomForestRegressor(n_estimators=100)

    regr.fit(X_train_max_sp, y_train_max_sp)

    sc = regr.score(X_test_max_sp, y_test_max_sp)
    print("r2_train: ", sc)




    params_for_maximize1 = ['Обводненность (вес), %', 'Время работы, ч', 'Забойное давление', 'Давление на приеме',
                          'Пластовое давление', 'Закачка, м3']
    X_test = test_data[params_for_maximize1]
    X_test = X_test.fillna(X_test.median())

    X_test['Забойное давление'] = X_test['Забойное давление'].apply(clean)

    X_test = X_test.fillna(X_test.median())

    doptest = X_test.copy()
    X_test = X_test.drop(['Закачка, м3'], axis=1)

    doptest['Нефть, т'] = regr.predict(X_test)

    doptest = doptest.fillna(doptest.median())

    doptest = doptest.sort_values('Нефть, т', ascending=False).loc[:num_left]

    final_xlsx =''
    # final_xlsx = doptest.to_excel('dd.xlsx')
    return doptest

d = Regression(cf.base_dir+cf.train_raw, cf.base_dir+cf.test_raw)

print(d.head())

#modes = SVM(d)
#print(modes)