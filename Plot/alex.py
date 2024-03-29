import matplotlib.pyplot as plt
import pandas as pd
from Hack import config as cf
import numpy as np

def Plot_jan_data(datafile: str):
    train_data = pd.read_excel(datafile)

    jan_data = train_data[train_data['Дата'] == pd.to_datetime('2016-10-01')]

    coordinates = pd.read_excel(cf.base_dir+cf.train_raw, sheet_name= 'Координаты')
    merged_data = pd.merge(jan_data, coordinates, how= 'left', left_on= 'Скважина', right_on= '№ скважины')

    # Plotting data
    plt.figure(figsize=(10,8))
    plt.scatter(x=merged_data['Координата X'], y= merged_data['Координата Y'], marker='o', c='r', edgecolor='b')

    for i in merged_data.index:
        plt.annotate(merged_data['Характер работы'][i], xy=(merged_data['Координата X'][i], merged_data['Координата Y'][i]))

    plt.savefig(cf.base_dir + '/WebPlot/WebPlot/static/images/Vyshki.png')
    # plt.show()

    return merged_data

Plot_jan_data(cf.base_dir+cf.train_raw)