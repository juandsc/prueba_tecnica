''' Implementation prueba tecnica Bancolombia - Prueba 2

Se define un experimento para determinar, a traves de una validacion cruzada,
cual es son los mejores parametro para creacion del modelo predictivo.
'''
import utils
import configparser
import pandas as pd
import json
import training_model as tm
import subprocess as sub
from sklearn import preprocessing

def get_params():
    '''Retorna los parametros que utiliza el script.

    # Retorna
        params: arreglo con los parametros del script
    '''
    config = configparser.ConfigParser()
    config.read('./params.ini')
    params = {}
    params['path_training_params'] = config['DEFAULT']['path_training_params']
    params['path_scheme_result'] = config['DEFAULT']['path_scheme_result']
    params['path_train'] = config['DEFAULT']['path_train']
    params['path_results'] = config['DEFAULT']['path_results']
    return params

def do_experiment():
    ''' Implementation del experimento a realizar.'''

    params = get_params()

    # Se prepara el dataset a utilizar.
    train = pd.read_csv(params['path_train'])
    y = train.FRAUDE.values
    train.drop('FRAUDE', axis=1, inplace=True)
    x = train.values
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)

    # Se cargan los parametros de entrenamiento
    with open(params['path_training_params']) as values:
        training_params = json.loads(values.read())

    # Se prepara el archivo donde se guardaran todos los resultados de las
    # validaciones.
    scheme_result = params['path_scheme_result']
    stats_file = params['path_results']
    sub.call(['cp', scheme_result, stats_file])

    # Se realiza una validacion para todas la combinaciones posibles de
    # parametros.
    for layers in training_params['layers']:
        for alpha in training_params['alpha']:
            for iters in training_params['iters']:
                for solver in training_params['solver']:
                    params_model = {'layers':layers, 'alpha':alpha,
                        'iters':iters, 'solver':solver}
                    stats_train, stats_test = \
                        tm.balanced_stratified_crossvalidation(x, y,
                        params_model)
                    utils.add_results(stats_train, stats_test, stats_file)

if __name__ == '__main__':
    do_experiment()
