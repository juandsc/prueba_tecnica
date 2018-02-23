''' Modulo para realizar las validacion de parametros.

La validaciones se realizan para una red neuronal
'''
import utils
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold



def balanced_stratified_crossvalidation(x, y, params):
    """ Realiza una valicacion cruzada estratificada balanceada.

    El numero predeterminado de folds es 5. Los parametros de entrenamiento
    estan dados por 'params'. La validacion se realiza para los conjuntos de
    entrenamiento y prueba. Esta estrategia intenta resolver el problema de
    desbalance de clase dividiendo la muestra negativas o no fraudulentas en
    n sub-grupos de tamano aproximadamente igual al numero de muestras
    positivas o fraudulentas. Se crean entonces n modelos, entrenados cada uno
    con el mismo numero de muestras positivas y negativas (todas las muestras
    positivas y un grupo de las muestras negativas). La prediccion final para
    una muestra sera el promedio de todas las predicciones de los n modelos.
    by the n models.

    # Argumentos
        x: muestras de entrada.
        y: salidas para cada muestra.
        params: parametros de entrenamiento.

    # Retornos
        stats_train: estadisticas de la validacion cruzada para conjunto de
        entrenamiento.
        stats_test: estadisticas de la validacion cruzada para conjunto de
        prueba.
    """

    stats_train = []
    stats_test = []

    fold = 1
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(x, y):
        init_time = time.time()
        print('--- Training Model: Fold '+str(fold)+' ---')

        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        # Divide los conjuntos de entrenamiento y prueba en sub-grupos positivos
        # y negativos. La palabra 'true' en las variables significa que el
        # sub-grupo son las salidas deseadas. La letra 'n' en las variables
        # significa que todas las muestras en el sub-grupo son negativas. La
        # letra 'p' en las variables significa que todas las muestras en el
        # sub-grupo son positivas.
        sets_training = utils.divide_by_labels(x_train, y_train)
        sets_testing = utils.divide_by_labels(x_test, y_test)
        x_ptrain, y_true_ptrain, x_ntrain, y_true_ntrain = sets_training
        x_ptest, y_true_ptest, x_ntest, y_true_ntest = sets_testing

        # Los conjuntos negativos de entrenamiento son divididos en sub-grupos
        # de tamano aproximadamente igual al tamano de los positivos.
        bands_train = utils.divide_negatives(x_ntrain, y_true_ntrain,
            x_ptrain.shape[0])

        # Listas para guardar las predicciones de todos los modelos para los
        # conjuntos de entrenamiento y prueba.
        predicts_ptrain = []
        predicts_ntrain = []
        predicts_ptest = []
        predicts_ntest = []

        # Por la forma en que fue dividido el conjunto de muestras negativas
        # el orden inicial de las salidas se pierde. Con este arreglo se
        # reconstruye el orden para poder calcular la medida de rendimiento con
        # sus correspondientes predicciones.
        y_true_ntrain = []
        for band in bands_train:
            # Reconstruccion parcial de 'y_true_ntrain'
            y_true_ntrain.extend(band[1])

            # Crea conjuntos de entrenamiento y prueba con la banda actual y el
            # conjunto de entrenamiento positivo.
            input_train = np.append(band[0], x_ptrain, axis=0)
            output_train = np.append(band[1], y_true_ptrain, axis=0)

            # creacion del modelo
            clf = MLPClassifier(solver=params['solver'], alpha=params['alpha'],
                hidden_layer_sizes=tuple(params['layers']),
                max_iter=params['iters'], random_state=1)

            clf.fit(input_train, output_train)

            # Acumula la predicciones actuales para todos los sub-grupos.
            predicts_ntrain.append(clf.predict(x_ntrain))
            predicts_ptrain.append(clf.predict(x_ptrain))
            predicts_ntest.append(clf.predict(x_ntest))
            predicts_ptest.append(clf.predict(x_ptest))

        # Calcula la prediccion final para cada sub-grupo y se juntan los
        # resultados para entrenamiento y prueba.
        pred_train = utils.join_results(predicts_ptrain,predicts_ntrain)
        pred_test = utils.join_results(predicts_ptest,predicts_ntest)

        # Se reconstruyen los conjuntos para las salidas de entrenamiento y
        # prueba.
        true_train = np.append(y_true_ptrain, y_true_ntrain)
        true_test = np.append(y_true_ptest, y_true_ntest)

        # Se calcula la medida AUC para cada grupo.
        fpr, tpr, thresholds = metrics.roc_curve(true_train, pred_train, pos_label=1)
        auc_train = metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(true_test, pred_test, pos_label=1)
        auc_test = metrics.auc(fpr, tpr)

        # Se juntan las estadisticas para cada grupo.
        train_data = ['train', fold, params['layers'], params['iters'],
            params['alpha'], params['solver']]
        test_data = ['test', fold, params['layers'], params['iters'],
            params['alpha'], params['solver']]

        # Se agrega el valor AUC a cada grupo de estadisticas.
        train_data.append(auc_train)
        test_data.append(auc_test)

        stats_train.append(train_data)
        stats_test.append(test_data)

        fold += 1
        finish_time = time.time()-init_time
        print('   --- Spended Time: '+str(finish_time/60))

    return stats_train, stats_test
