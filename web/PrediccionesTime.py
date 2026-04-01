"""
Docstring per web.PrediccionesTime
"""
import joblib
import tensorflow as tf
import datetime

from tensorflow.keras.models import load_model

from processtransformer.constants import Task
from FuncionesAuxiliares import *
from ConfigLoader import get_processing_dir, get_staging_dir
from processtransformer.models.transformer import TokenAndPositionEmbedding, TransformerBlock


class PrediccionesNextTime:
    def __init__(self):
        self.atributo1 = "sdf"

    def prepare_predict_next_time(self, nombreProceso,nombreLog,nombreModelo,nombreTipoPred, nombreTraza):

        ruta_procesos = get_processing_dir()
        ruta_proceso = os.path.join(ruta_procesos, nombreProceso)
        ruta_carpeta_log = os.path.join(ruta_proceso, nombreLog)
        ruta_tipoPred = os.path.join(ruta_carpeta_log, nombreTipoPred)
        rutaModelo = os.path.join(ruta_tipoPred, nombreModelo)
        rutapropiedades = os.path.join(rutaModelo, 'propiedadesModelo.txt')

        ruta_trazas = get_staging_dir()

        # Carga las propiedades relacionadas al modelo

        propiedades = load_variables(rutapropiedades)
        x_word_dict = propiedades[0]
        y_word_dict = propiedades[1]
        max_case_length = propiedades[2]
        columnID = propiedades[3]
        columnT = propiedades[4]
        column_pred = propiedades[5]
        inputTask = propiedades[8]
        formatoT = propiedades[10]
        participant = propiedades[11]

        path_traza = os.path.join(ruta_trazas, nombreTraza)
        columnas_traza = obtener_nombre_columnas(path_traza)

        # Busca la columna de Prediccion
        if (column_pred not in columnas_traza):
           raise ValueError(f"La columna '{column_pred}' no se encuentra en el archivo.")

        index_pred = find_column_index(os.path.join(ruta_trazas, nombreTraza), column_pred)
        column_pred_values = extract_column(os.path.join(ruta_trazas, nombreTraza), index_pred)
        # Reemplazar espacios por "-" y convertir a minúsculas
        column_pred_values = [s.replace(' ', '-').lower() for s in column_pred_values]

        # Busca la columna de Tiempo
        if (columnT not in columnas_traza):
           raise ValueError(f"La columna '{columnT}' no se encuentra en el archivo.")

        index_time = find_column_index(os.path.join(ruta_trazas, nombreTraza), columnT)
        column_time_values = extract_column(os.path.join(ruta_trazas, nombreTraza), index_time)

        # Busca la columna de id del caso
        if (columnID not in columnas_traza):
            raise ValueError(f"La columna '{columnID}' no se encuentra en el archivo.")

        index_caseid = find_column_index(os.path.join(ruta_trazas, nombreTraza), columnID)
        column_caseid_values = extract_column(os.path.join(ruta_trazas, nombreTraza), index_caseid)

        # identifico cuantos y cuales casos hay en el archivo cargado de trazas
        casos, indices = np.unique(column_caseid_values,return_index=True)
        casos = casos[np.argsort(indices)]

        current_caseid=column_caseid_values[0]
        ini = 0
        fin = 0
        token_x = np.array([])
        time_x = np.array([])
        ult_pos=len(column_caseid_values) #

        for c in column_caseid_values:

            if (c == current_caseid) and (ult_pos != fin+1): # avanzo mientras sea el mismo caso
                fin+=1
            else:
                if ult_pos != (fin+1): #si llego a un caso distinto y no es el final
                    column_pred_caseid = column_pred_values[ini:fin] #obtengo de la columna column_pred de la traza, los valores que corresponden al current_caseid
                    column_time_caseid = column_time_values[ini:fin]  # obtengo de la columna column_time de la traza, los valores que corresponden al current_caseid
                    ini = fin
                    fin+=1
                else: #si llego al final
                    column_pred_caseid = column_pred_values[ini:ult_pos+1]
                    column_time_caseid = column_time_values[ini:ult_pos + 1]

                token_x_fila = genera_fila_token_x(max_case_length, x_word_dict, column_pred_caseid) # genero token_x_fila

                if inputTask == Task.NEXT_TIME_MESSAGE:
                    time_x_fila = genera_fila_time_message_x(column_pred_caseid,column_time_caseid,formatoT)
                elif inputTask in (Task.NEXT_TIME,Task.REMAINING_TIME):
                    time_x_fila = genera_fila_time_x_originales(column_pred_caseid, column_time_caseid,formatoT)
                else:
                    time_x_fila = genera_fila_remaining_time_x(column_pred_caseid, column_time_caseid, participant, formatoT)

                if token_x.size == 0:
                    token_x = np.stack(token_x_fila)
                    time_x = np.stack(time_x_fila)
                else:
                    token_x = np.vstack([token_x, token_x_fila]) # agrego a la matriz token_x
                    time_x = np.vstack([time_x,time_x_fila])

                current_caseid=c

        # ----armo time_x
        time_x = np.array(time_x, dtype=np.float32)

        ruta_scalers = os.path.join(rutaModelo, "scalers")
        ruta_absoluta_y_scaler = os.path.join(ruta_scalers, "y_scaler.pkl")
        ruta_absoluta_time_scaler = os.path.join(ruta_scalers,"time_scaler.pkl")

        # ----obtengo time_scaler para invertir la prediccion
        time_scaler = joblib.load(ruta_absoluta_time_scaler)

        time_x = time_scaler.transform(time_x).astype(np.float32)

        time_x = np.array(time_x, dtype=np.float32)

        # ----armo token_x

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)

        token_x = np.array(token_x, dtype=np.float32)

        #----obtengo y_scaler para invertir la prediccion
        y_scaler = joblib.load(ruta_absoluta_y_scaler)
        #----

        # -------------------PREDICCION----------------

        ruta_nombreModeloTransformer = os.path.join(rutaModelo, nombreModelo + "T.keras")
        modelo = load_model(
            ruta_nombreModeloTransformer,
            custom_objects={
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "TransformerBlock": TransformerBlock,
            },
            compile=False,
        )

        y_pred = modelo.predict([token_x,time_x])

        _y_pred = y_scaler.inverse_transform(y_pred)

        #----Preparo la salida------
        caseid_pred=np.array([])
        index_caseid=0
        for y in _y_pred:

            arreglo_aux=[[casos[index_caseid],np.round(y,1)[0]]]

            if caseid_pred.size == 0:
                caseid_pred = np.stack(arreglo_aux)
            else:
                caseid_pred = np.vstack([caseid_pred, arreglo_aux])  # agrego a la matriz token_x

            index_caseid+=1
        
        # Sort predictions by case_id (natural sort for strings like "case_1", "case_10", etc.)
        import re
        def natural_sort_key(item):
            """Sort key function for natural sorting (handles numbers within strings)"""
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(item[0]))]
        
        caseid_pred = sorted(caseid_pred, key=natural_sort_key)
        caseid_pred = np.array(caseid_pred)
        
        caseid_pred = formato_salida_prediccion_tiempo(caseid_pred, formatoT)

        return caseid_pred,participant


def find_column_index(csv_file, column_name):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Lee la primera fila como encabezados

        try:
            index = headers.index(column_name)  # Obtiene el índice de la columna por nombre
            return index
        except ValueError:
            print(f'Columna "{column_name}" no encontrada en el archivo CSV.')

def extract_column(csv_file, column_index):
    column_values = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Salta la primera fila (encabezados)
        for row in reader:
            if len(row) > column_index:
                column_values.append(row[column_index])
    return column_values

def genera_fila_token_x(max_case_length, x_word_dict,column_values):

    token_x = np.zeros((1, max_case_length))

    for c in column_values:
        val = x_word_dict.get(c.lower())
        token_x[0] = np.insert(token_x[0][1:], token_x[0].size - 1, val)

    return token_x

def genera_fila_time_x_originales(act, time,formatoT):  # prediccion de tiempo de proxima actividad, act: actividades de un caso_x

    time_passed = 0
    latest_diff = datetime.timedelta()
    recent_diff = datetime.timedelta()

    for i in range(0, len(act)):

        if i > 0:
            latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                          datetime.datetime.strptime(time[i - 1], "%Y-%m-%d %H:%M:%S")



        latest_time = np.where(i == 0, 0, int((latest_diff.total_seconds() // formatoT)))

                # -----------------
        if i > 1:
            recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                          datetime.datetime.strptime(time[i - 2], "%Y-%m-%d %H:%M:%S")


        recent_time = np.where(i <= 1, 0, int((recent_diff.total_seconds() // formatoT)))
        # -----------

        time_passed_aux = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                          datetime.datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")

        time_passed = int((time_passed_aux.total_seconds() // formatoT))

        # --------

    # ---me quedo con el ultimo generado
    fila_time_x = [[recent_time.item(), latest_time.item(), time_passed]]
    # se le agrega .item() porque son arrays y queremos solo el valor



    return fila_time_x

def genera_fila_time_message_x(send, time,formatoT):  # prediccion de tiempo del proximo mensaje, act: actividades de un caso_x

        time_passed = 0
        latest_diff = datetime.timedelta()
        recent_diff = datetime.timedelta()
        time_message = []
        z=0


        for i in range(0, len(send)):
            prefix = np.where(i == 0, send[0], " ".join(send[:i + 1]))

            # LASTEST_TIME -------------------------
            if z > 0:
                latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(time_message[z - 1], "%Y-%m-%d %H:%M:%S")

            latest_time = np.where(z == 0, 0, int((latest_diff.total_seconds() // formatoT)))

            # RECENT_TIME ---------------------------
            if z > 1:  # esta comparacion seria en el arreglo de "sendtask"
                recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(time_message[z - 2], "%Y-%m-%d %H:%M:%S")

            recent_time = np.where(z <= 1, 0, int((recent_diff.total_seconds() // formatoT)))

            # TIME_PASSED---------------------------

            time_passed_aux = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")

            time_passed = int((time_passed_aux.total_seconds() // formatoT))

            # Si es un nuevo send, lo agrego a la lista
            if send[i].__contains__("sendtask"):
                time_message.insert(z, time[i])
                z = z + 1

        #---me quedo con el ultimo generado
        fila_time_message_x= [[recent_time.item(),latest_time.item(),time_passed]]
        # se le agrega .item() porque son arrays y queremos solo el valor

        return fila_time_message_x

def genera_fila_remaining_time_x(part, time,participant,formatoT):

        time_passed = 0
        latest_diff = datetime.timedelta()
        recent_diff = datetime.timedelta()

        time_participant = []
        z = 0

        for i in range(0, len(part)):
            #
            prefix = np.where(i == 0, part[0], " ".join(part[:i + 1]))

            # LASTEST_TIME -------------------------
            if z > 0:
                latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(time_participant[z - 1], "%Y-%m-%d %H:%M:%S")

            latest_time = np.where(z == 0, 0, int((latest_diff.total_seconds() // formatoT)))

            # RECENT_TIME ---------------------------
            if z > 1:  # esta comparacion seria en el arreglo de "sendtask"
                recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(time_participant[z - 2], "%Y-%m-%d %H:%M:%S")

            recent_time = np.where(z <= 1, 0, int((recent_diff.total_seconds() // formatoT)))

            # TIME_PASSED---------------------------

            time_passed_aux = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")

            time_passed = int((time_passed_aux.total_seconds() // formatoT))

            # si el participante es el elegido, agrego tiempos a arreglo auxiliar
            if part[i].__contains__(participant):
                time_participant.insert(z, time[i])
                z = z + 1

        # ---me quedo con el ultimo generado
        fila_time_remaining_time_x = [[recent_time.item(), latest_time.item(), time_passed]]
        # se le agrega .item() porque son arrays y queremos solo el valor

        return fila_time_remaining_time_x