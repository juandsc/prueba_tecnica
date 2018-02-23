import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

def do_prediction():
    train = pd.read_csv('../data_processing/new_train.csv')
    test = pd.read_csv('../data_processing/new_test.csv')

    test_initial = pd.read_csv('../data_processing/test2.csv')

    y_train = train.FRAUDE.values
    train.drop('FRAUDE', axis=1, inplace=True)
    x_train = train.values
    scaler1 = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler1.transform(x_train)

    test.drop('FRAUDE', axis=1, inplace=True)
    x_test = test.values
    scaler2 = preprocessing.StandardScaler().fit(x_train)
    x_test = scaler2.transform(x_test)

    clf = MLPClassifier(solver='adam', alpha=0.1,
        hidden_layer_sizes=(124,62,31,),random_state=1,max_iter=200)
    clf.fit(x_train, y_train)

    predicts = clf.predict(x_test)
    test_initial.FRAUDE = predicts.astype(int)
    test_initial.to_csv('./test_evaluado.csv', index=False)


if __name__ == '__main__':
    do_prediction()
