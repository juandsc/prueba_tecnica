''' Modelo con metodo que se utilizan en el entrenamiento de los modelos.'''

import numpy as np
import pandas as pd

def divide_by_labels(inputs, outputs):
    """ Divide las entradas y salidas en grupos en sub-grupos.

    La division se realiza segun sea la salida de cada muestra..

    # Argumentos
        inputs: muestras de entrada.
        outputs: salidas de las muestras.

    # Retornos
        pos_inputs: muestras positivas de entrada.
        pos_outputs: salidas positivas de las muestras.
        neg_inputs: muestras negativas de entrada.
        neg_outputs: salidas negativas de las muestras.
    """
    positives_indexes = []
    for i in range(inputs.shape[0]):
        if outputs[i] == 1:
            positives_indexes.append(i)
    pos_inputs = inputs[positives_indexes]
    pos_outputs = outputs[positives_indexes]
    neg_inputs = np.delete(inputs, positives_indexes, axis=0)
    neg_outputs = np.delete(outputs, positives_indexes, axis=0)
    return pos_inputs, pos_outputs, neg_inputs, neg_outputs


def divide_negatives(inputs, outputs, num_pos):
    """ Divide las entradas y salidas en n bandas.

    Se divide 'inputs' y 'outputs' en n bandas de tamano aproximadamente igual
    a 'num_pos'. Se crea una lista donde cada elemento en una tupla, cada una
    con un sub-grupo de 'inputs' y otro de 'outputs'.

    # Argumentos
        inputs: muestras de entrada negativas.
        outputs: salidas para las muestras negativas.

    # Returns
        bands: lista con todas las bandas para 'inputs' y 'outputs'.
    """
    num_inputs = inputs.shape[0]
    num_groups = int(num_inputs/num_pos)
    addition = int((num_inputs - (num_groups * num_pos)) / num_groups)
    group_size = num_pos + addition
    bands = []
    for i in range(1, num_groups+1):
        i_index = i*group_size
        i_band_input = inputs[i_index-group_size:i_index]
        i_band_output = outputs[i_index-group_size:i_index]
        bands.append([i_band_input, i_band_output])
    n = 0
    for j in range(i_index, num_inputs):
        bands[n][0] = np.append(bands[n][0], inputs[j:j+1], axis=0)
        bands[n][1] = np.append(bands[n][1], outputs[j:j+1], axis=0)
        n += 1
    return bands


def join_results(pred_pos, pred_neg):
    """ Calcula la prediccion final para dos conjuntos de muestras.

    'pred_pos' y 'pred_neg' son todas las predicciones hechas por n modelos
    para muestras positivas y negativas, correspondientemente. Se calcula la
    media de las predicciones y se redondea

    # Argumentos
        pred_pos: lista de predicciones para muestras positivas.
        pred_neg: lista de predicciones para muestras negativas.

    # Retornos
        labels: predicciones finales agrupadas de los dos conjuntos.
    """
    pred_pos = np.array(pred_pos)
    pred_neg = np.array(pred_neg)
    labels_pos = np.round(pred_pos.T.mean(axis=1))
    labels_neg = np.round(pred_neg.T.mean(axis=1))
    labels = np.append(labels_pos, labels_neg, axis=0)
    return labels


def get_grouped_measures(measures):
    """ Agrupa todas las medidas calculadas en una validacion cruzada.

    Despues de una validacion cruzada se agrupan los parametros de
    entrenamiento y se calcula la media para AUC.

    # Argumentos
        measures: Lista con todas la medidas una validacion cruzada.

    # Retornos
        parameters: Lista con las medidas agrupadas.
    """

    parameters = measures[0][:6]
    auc_mean = np.array([i[6] for i in measures]).mean()
    parameters.append(auc_mean)
    return parameters


def add_to_dataframe(data, table_path, sheet):
    """ Agrega los resultados de una validacion a la tabla de resultados.

    La tabla tiene los resultados de todas las validaciones cruzadas. Esta
    tabla tiene dos hojas, una con todos los resultados de los folds y otra
    con los valores agrupados para cada validacion.

    # Argumentos
        data: resultados de una validacion cruzada.
        table_path: ruta a la tabla que tiene todos los resultados.
        sheet: indice de la hoja que se quire actualizar. 0 para la hoja con
        todos los resultados y 1 para los resultados agrupados.
    # Retornos:
        Dataframe con los resultados agregados.
    """
    data_frame = pd.read_excel(table_path, sheet_name=sheet)
    pd_data = pd.DataFrame(data, columns=data_frame.columns.values)
    return data_frame.append(pd_data)


def save_results(measures, grouped, table_path):
    """ Guarda los resultados de una validacion cruzada.

    Los resultados se guardan en una tabla que tiene todos los resultados. La
    estructura de la tabla fue definida previamente.

    # Argumentos
        measures: Lista con todos los resultados de la validacion cruza.
        grouped: Lista con los resultados agrupados.
        table_path: ruta a la tabla con todos los resultados.
    """
    pd_measures = add_to_dataframe(measures, table_path, 0)
    pd_grouped = add_to_dataframe(grouped, table_path, 1)
    writer = pd.ExcelWriter(table_path, engine='openpyxl')
    pd_measures.to_excel(writer, 'all', index=False)
    pd_grouped.to_excel(writer, 'means', index=False)
    writer.save()


def add_results(measures_train, measures_test, file_path):
    """ Coordina la agregacion de los resultados de una validacion cruzada.

    Los resultados pertenecen a conjuntos de entrenamiento y prueba. Se guardan
    en un archivo que tiene todos los resultados.

    # Argumentos
        measures_train: lista de resultados para el conjunto de entrenamiento.
        measures_test: lista de resultados para el conjunto de prueba.
        file_path: ruta a la tabla con todos los resultados.
    """
    grouped_train = get_grouped_measures(measures_train)
    grouped_test = get_grouped_measures(measures_test)
    measures_train.extend(measures_test)
    grouped = [grouped_train, grouped_test]
    save_results(measures_train, grouped, file_path)
