# Предсказыает режим работы (НЕФ/НАГ) в зависимости от значения "Закачка, м3"
from sklearn import svm
import pandas as pd
import config as cf
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

def SVM():
    data = pd.read_excel(cf.base_dir+cf.train_raw)
    test_data = pd.read_excel(cf.base_dir+cf.test_raw)


    data = data[data['Время работы, ч'] != 0].fillna(0)
    data = data[data['Характер работы'] != 'НЕФ/НАГ']

    test_data = test_data[test_data['Время работы, ч'] != 0].fillna(0)
    test_data = test_data[test_data['Характер работы'] != 'НЕФ/НАГ']

    lb = LabelBinarizer()
    X_train = data[["Закачка, м3"]]
    y_train = data['Характер работы']
    y_train = lb.fit_transform(y_train)

    X_test_data = test_data[["Закачка, м3"]]




    X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X_train, y_train, test_size=0.3)
    print('SVM is fitting......')
    model_svc = svm.SVC(verbose=10)
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

SVM()