import pandas as pd
import config as cf

def RandomForest():
    train_data = pd.read_csv(cf.base_dir+cf.train_raw, low_memory=False).loc[:2000]

    X = train_data[['Нефть, м3', 'Жидкость, м3', 'Забойное давление', 'Обводненность']]


    '''
    X_sorted = X.sort_values(by="Глубина верхних дыр перфорации", ascending=False)
    val_data = pd.value_counts(X_sorted.values.flatten(), ascending=True)
    a = X_sorted.loc[X_sorted['Глубина верхних дыр перфорации']== 3391.0]
    '''
    print(X)

RandomForest()