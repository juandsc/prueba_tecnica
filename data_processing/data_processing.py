''' Preprocesamiento de datos.

Se realiza las modificaciones necesarias al dataset de modo que pueda ser
utilizado por el modelo predictivo.
'''
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import preprocessing


def impute_mode(data_frame, columns):
    for column in columns:
        mode = data_frame[column].value_counts().index[0]
        data_frame.loc[data_frame[column].isnull(), column] = mode
    return data_frame


def impute_mean(data_frame, columns):
    for column in columns:
        mean = data_frame[column].mean()
        data_frame.loc[data_frame[column].isnull(), column] = mean
    return data_frame


def impute_categorical_values(categories):
    nulls = categories.isnull().any()
    columns_null = nulls.loc[nulls == True].index.values
    new_categories = impute_mode(categories, columns_null)
    return new_categories


def impute_continuous_values(continuous):
    nulls = continuous.isnull().any()
    columns_null = nulls.loc[nulls == True].index.values
    new_continuous = impute_mean(continuous, columns_null)
    return new_continuous


def do_processing(data):
    ''' Realiza las transformaciones e imputaciones necesarias.'''

    # Seleccion de variables categoricas e imputacion de valores faltantes
    categories = data.select_dtypes(include=[object])
    categories = impute_categorical_values(categories)
    data.drop(categories.columns.values, axis=1, inplace=True)

    # Definicion de funcion que transforma una fecha a su valor timestamp.
    to_timestamp = lambda x: time.mktime(datetime.datetime.strptime(str(int(x)), "%Y%m%d").timetuple())
    # Transformacion de columnas fecha a su valor timestamp.
    condition = (data.FECHA.notnull())
    data.loc[condition, 'FECHA'] = list(map(to_timestamp, data.loc[condition]['FECHA']))
    condition = (data.FECHA_VIN.notnull())
    data.loc[condition, 'FECHA_VIN'] = list(map(to_timestamp, data.loc[condition]['FECHA_VIN']))

    # Imputacion de valores faltantes a variables continuas.
    data = impute_continuous_values(data)

    # Variable categorica con valor numerico que debe ser imputada
    data = impute_mode(data, ['OFICINA_VIN'])

    # Transformacion de variables categoricas a one hot encoding.
    le = LabelEncoder()
    categories = categories.apply(le.fit_transform)
    enc = OneHotEncoder()
    enc.fit(categories)
    onehotlabels = enc.transform(categories).toarray()
    df_onehotlabels = pd.DataFrame(onehotlabels)
    # Se juntan todas las caracteriscas.
    data = data.join(df_onehotlabels)
    return data

if __name__ == '__main__':
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test2.csv')
    data = train.append(test, ignore_index=True)
    new_data = do_processing(data)
    new_train = new_data.iloc[:train.shape[0]]
    new_test = new_data.iloc[train.shape[0]:]
    new_train.to_csv('./new_train.csv', index=False)
    new_test.to_csv('./new_test.csv', index=False)
