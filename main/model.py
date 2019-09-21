import pandas as pd
import config as cf
from sklearn.linear_model import LogisticRegression
from Plot.alex import Plot_jan_data
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import numpy as np
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def LogReg():

    # Loading Data
    #train_data = Plot_jan_data(cf.base_dir+cf.train_raw)
    train_data = pd.read_excel(cf.base_dir+cf.train_raw)
    print("Train, not processed: ", train_data.shape)
    train_data = train_data[train_data['Время работы, ч'] != 0]
    train_data = train_data[['Нефть, м3', 'Жидкость, м3', 'Забойное давление', 'Обводненность (вес), %', 'Нефть, т', 'Характер работы']]
    train_data = train_data.fillna(0)
    print('Train processed: ', train_data.shape)

    test_data = Plot_jan_data(cf.base_dir+cf.test_raw)
    test_data = test_data[['Нефть, м3', 'Жидкость, м3', 'Забойное давление', 'Обводненность (вес), %', 'Нефть, т', 'Характер работы']]
    test_data = test_data.fillna(0)

    parameters = ['Попутный газ, м3', ]
    X_train = train_data[['Нефть, м3', 'Жидкость, м3', 'Обводненность (вес), %']]
    y_train = train_data[['Нефть, т']].astype(int)

    X_test = train_data[['Нефть, м3', 'Жидкость, м3', 'Обводненность (вес), %']]
    y_test = train_data[['Нефть, т']].astype(int)

    # Set Classifier
    clf = LogisticRegression(C=0.01, penalty='l1')
    '''
    # Grid Search
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    logreg_cv = RandomizedSearchCV(clf, grid, cv=2, n_iter=3)
    logreg_cv.fit(X_train, y_train)

    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)
    '''
    # Fitting Logistic Regression to the Training set
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    '''
    
    X_sorted = X.sort_values(by="Глубина верхних дыр перфорации", ascending=False)
    val_data = pd.value_counts(X_sorted.values.flatten(), ascending=True)
    a = X_sorted.loc[X_sorted['Глубина верхних дыр перфорации']== 3391.0]
    '''


LogReg()