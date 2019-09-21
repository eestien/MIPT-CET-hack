import pandas as pd
import config as cf
import matplotlib.pyplot as plt


def RandomForest():
    coordinates = pd.read_excel(cf.base_dir+cf.train_raw, sheet_name="Координаты")
    train_data = pd.read_excel(cf.base_dir+cf.train_raw, sheet_name="Месторождение 1")



    skv = train_data['Скважина'][train_data['Дата'] == pd.to_datetime('2016-12-01')]
    skv = skv.tolist()

    selected_coordinates = coordinates[['Координата X', 'Координата Y']][coordinates['№ скважины'].isin(skv)]
    # day_year = train_data[train_data['Дата'] == pd.to_datetime('2016-09-01')]
    print(type(selected_coordinates))

    # day_and_skv = pd.merge(day_year, skv)
    # selected_coordinates = pd.merge()
    print(selected_coordinates.shape)

    X = selected_coordinates['Координата X'].tolist()
    Y = selected_coordinates['Координата Y'].tolist()

    '''
    # Plotting 
    plt.xlabel("X", fontsize=10)
    plt.ylabel("Y", fontsize=10)
    plt.scatter(X, Y, s=5)
    for i in df3.index:
        plt.annotate(df3['Характер работы'][i], xy=(df3['Координата X'][i], df3['Координата Y'][i]))

    plt.show()
    plt.show()
    '''

RandomForest()