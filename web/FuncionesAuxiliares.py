"""
Docstring per web.FuncionesAuxiliares
"""
import os
import pickle
import csv
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ConfigLoader import get_processing_dir, get_max_decimal_places


def filter_system_files(file_list):
    """
    Filter out system files and hidden files from a list.
    Removes files like .DS_Store, .gitkeep, Thumbs.db, and any file starting with '.'
    
    Args:
        file_list: List of filenames or directory names
        
    Returns:
        Filtered list without system files
    """
    system_files = {'.DS_Store', '.gitkeep', 'Thumbs.db', 'desktop.ini', '.localized'}
    return [f for f in file_list if f not in system_files and not f.startswith('.')]


def save_variables(filename, *args):
    with open(filename, 'wb') as file:
        for var in args:
            pickle.dump(var, file)

def save_model_metrics_to_csv(log_folder_path, process_name, log_file_name, model_name, prediction_type, metric1, metric2, metric3, metric4):
    """
    Save model metrics to a CSV file in the log folder.
    All metrics columns are always present; non-applicable metrics are marked with -1.
    Metrics are rounded according to max_decimal_places from config.yml.
    
    Args:
        log_folder_path: Path to the log folder (e.g., processing/processo/log)
        process_name: Name of the process
        log_file_name: Name of the log file
        model_name: Name of the model
        prediction_type: Type of prediction (NextActivity, ParticipantSend, etc.)
        metric1-4: Metric values (names depend on prediction type)
    """
    csv_file_path = os.path.join(log_folder_path, 'models_metrics.csv')
    file_exists = os.path.exists(csv_file_path)
    
    # Get decimal places from config
    decimal_places = get_max_decimal_places()
    
    # Helper function to round metrics
    def round_metric(value):
        if value is None or value == -1:
            return -1
        return round(float(value), decimal_places)
    
    # Define all possible columns in a fixed order
    fieldnames = ['Model_Name', 'Timestamp', 'Process', 'File', 'Prediction_Type', 
                  'Accuracy', 'F-Score', 'Precision', 'Recall',
                  'MAE', 'MSE', 'RMSE']
    
    # Prepare row data with all metrics
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # For classification tasks (activity/participant prediction)
    if prediction_type in ["NextActivity", "ParticipantSend", "ActivityParticipantSend", 
                           "ActivityParticipant", "Participant"]:
        row_data = {
            'Timestamp': timestamp,
            'Process': process_name,
            'File': log_file_name,
            'Model_Name': model_name,
            'Prediction_Type': prediction_type,
            'Accuracy': round_metric(metric1),
            'F-Score': round_metric(metric2),
            'Precision': round_metric(metric3),
            'Recall': round_metric(metric4),
            'MAE': -1,
            'MSE': -1,
            'RMSE': -1
        }
    # For regression tasks (time prediction)
    else:
        row_data = {
            'Timestamp': timestamp,
            'Process': process_name,
            'File': log_file_name,
            'Model_Name': model_name,
            'Prediction_Type': prediction_type,
            'Accuracy': -1,
            'F-Score': -1,
            'Precision': -1,
            'Recall': -1,
            'MAE': round_metric(metric1),
            'MSE': round_metric(metric2),
            'RMSE': round_metric(metric3)
        }
    
    # Write to CSV
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)

def entrada_valida(entrada):
    patron = r'^[a-zA-Z0-9-_]+$'

    if re.match(patron, entrada):
        return True
    else:
        return False

def load_variables(filename):
    loaded_vars = []
    with open(filename, 'rb') as file:
        while True:
            try:
                var = pickle.load(file)
                loaded_vars.append(var)
            except EOFError:
                break
    return loaded_vars


def obtener_nombre_columnas(log_path):
  # Inicializar una lista para los valores de la primera fila
    nombres_columna = []
   # Abrir el archivo CSV
    with open(log_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Leer la primera fila del log
        primera_fila = next(reader, None)
        if primera_fila is not None:
            nombres_columna = primera_fila

    return nombres_columna

def obtener_columnas_por_proceso_y_log(proceso, log):

    ruta_procesos = get_processing_dir()

    archivo_propiedades = ruta_procesos +"/" + proceso + "/" + log + "/" + "propiedadesLog.txt"

    propiedades=load_variables(archivo_propiedades)

    columnas = propiedades[1]
    return columnas

def concatenate_columns(input_csv, col1, col2, col3):

    # Leer el archivo CSV
    df = pd.read_csv(input_csv)

    if col3 == None:
        # Crear el nombre de la nueva columna
        new_col_name = f"{col1}_{col2}"

        # Verificar si las columnas existen en el DataFrame
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Una o ambas columnas '{col1}' o '{col2}' no se encuentran en el archivo CSV.")

        if new_col_name in df.columns:
            raise ValueError(f"La columna '{new_col_name}' ya existe en el archivo CSV.")

        # Concatenar las columnas especificadas con un guion bajo
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
    else:
        # Crear el nombre de la nueva columna
        new_col_name = f"{col1}_{col2}_{col3}"

        # Verificar si las columnas existen en el DataFrame
        if col1 not in df.columns or col2 not in df.columns or col3 not in df.columns:
            raise ValueError(f" Alguna de las columnas '{col1}', '{col2}' o '{col3}'no se encuentran en el archivo CSV.")

        if new_col_name in df.columns:
            raise ValueError(f"La columna '{new_col_name}' ya existe en el archivo CSV.")

        # Concatenar las columnas especificadas con un guion bajo
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str) + '_' + df[col3].astype(str)

    # Guardar el DataFrame modificado en el mismo archivo CSV
    df.to_csv(input_csv, index=False)

    return new_col_name

def formato_salida_prediccion(caseid_pred,tipoPred):
    caseid_pred2 = np.array([])
    # Asignamos el texto dependiendo del valor recibido
    for pred in caseid_pred:
        aux1 = pred[1].replace("-", " ")  # send-task_laboratory -> send task_laboratory
        aux2 = aux1.split("_")  # send task_laboratory -> [send task,laboratory] send-blood-sample_gynecologist
        if tipoPred == "NextActivity":
            arreglo_aux = [[pred[0], aux1]]
        elif tipoPred == "Participant":
            arreglo_aux = [[pred[0], aux1]]
        elif tipoPred == "ActivityParticipant":
            valor="Next activity: " + aux2[0] + " - Participant: " + aux2[1]
            arreglo_aux = [[pred[0], valor]]
        elif tipoPred == "ParticipantSend":
            arreglo_aux = [[pred[0], aux2[1]]]
        elif tipoPred == "ActivityParticipantSend":
            valor = "Next message: " + aux2[1] + " - From participant: " + aux2[2]
            arreglo_aux = [[pred[0], valor]]

        if caseid_pred2.size == 0:
            caseid_pred2 = np.stack(arreglo_aux)
        else:
            caseid_pred2 = np.vstack([caseid_pred2, arreglo_aux])

    return caseid_pred2

def formato_salida_prediccion_tiempo(caseid_pred,formatoT):

    if formatoT == 1:
        formatoSalida = " Seconds"
    elif formatoT == 60:
        formatoSalida = " Minutes"
    elif formatoT == 3600:
        formatoSalida = " Hours"
    elif formatoT == 86400:
        formatoSalida = " Days"

    caseid_pred2 = np.array([])
    for pred in caseid_pred:
        if abs(float(pred[1])) < 1:
            aux="0" + formatoSalida
        else:
            aux=pred[1] + formatoSalida
        arreglo_aux = [[pred[0], aux]]
        if caseid_pred2.size == 0:
            caseid_pred2 = np.stack(arreglo_aux)
        else:
            caseid_pred2 = np.vstack([caseid_pred2, arreglo_aux])  # agrego a la matriz token_x

    return caseid_pred2


def listar_logs():
    ruta_procesos = get_processing_dir()

    # Processing directory is created at application startup
    # if not os.path.exists(ruta_procesos):
    #     os.mkdir(ruta_procesos)

    logs = []
    # Recorrer la estructura de directorios
    for proceso in filter_system_files(os.listdir(ruta_procesos)):
        proceso_path = os.path.join(ruta_procesos, proceso)
        if os.path.isdir(proceso_path):
            for log in filter_system_files(os.listdir(proceso_path)):
                log_path = os.path.join(proceso_path, log)
                if os.path.isdir(log_path):
                    prop=load_variables(os.path.join(log_path,"propiedadesLog.txt"))
                    descripcionLog = prop[0]
                    nombreProceso = prop[2]
                    logs.append({
                        'nombreProceso': nombreProceso,
                        'log': log,
                        'descripcionLog': descripcionLog
                    })
    return logs

def existen_modelos(proceso,log):

    folder_path_log = os.path.join(get_processing_dir(), proceso, log)

    tiposPred = []
    for tipoPred in filter_system_files(os.listdir(folder_path_log)):
        tiposPred_path = os.path.join(folder_path_log, tipoPred)
        if os.path.isdir(tiposPred_path):
            tiposPred.append(tipoPred)

    if not tiposPred:
        return False
    else:
        return True