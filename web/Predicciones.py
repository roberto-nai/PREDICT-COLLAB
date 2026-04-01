"""
Docstring per web.Predicciones
"""
from tensorflow.keras.models import load_model

from Modelos import *
from ConfigLoader import get_processing_dir, get_staging_dir
from processtransformer.models.transformer import TokenAndPositionEmbedding, TransformerBlock


class Predicciones:
    def __init__(self):
        self.atributo1 = "sdf"

    def prepare_predict(self, nombreProceso,nombreLog,nombreModelo,nombreTipoPred, nombreTraza):

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
        column1 = propiedades[5]
        column2 = propiedades[6]
        column3 = propiedades[7]
        tipoPred = propiedades[9]

        path_traza=os.path.join(ruta_trazas, nombreTraza)
        columnas_traza=obtener_nombre_columnas(path_traza)

        #creo la columna en la traza
        column_pred = ""
        if tipoPred in ("NextActivity", "Participant"):
            if (column1 in columnas_traza):
                column_pred = column1
            else:
                raise ValueError(f"La columna '{column1}' no se encuentra en el archivo.")
        elif tipoPred in ("ParticipantSend", "ActivityParticipant"):
            if (column1 in columnas_traza) and (column2 in columnas_traza):
                column_pred = concatenate_columns(os.path.join(ruta_trazas, nombreTraza), column1, column2, None)
            else:
                raise ValueError(f"La columna '{column1}' o '{column2}' no se encuentra en el archivo.")
        elif tipoPred == "ActivityParticipantSend":
            if (column1 in columnas_traza) and (column2 in columnas_traza) and (column3 in columnas_traza):
                column_pred = concatenate_columns(os.path.join(ruta_trazas, nombreTraza), column1, column2, column3)
            else:
                raise ValueError(f"La columna '{column1}','{column2}' o '{column3}' no se encuentra en el archivo.")

        # Busca la columna para predicción
        index_pred= find_column_index(path_traza,column_pred)
        column_pred_values = extract_column(path_traza,index_pred)
        # Reemplazar espacios por "-" y convertir a minúsculas
        column_pred_values = [s.replace(' ', '-').lower() for s in column_pred_values]

        #valida si existe la columna id en la taza
        if (columnID in columnas_traza):
            # Busca la columna de id del caso
            index_caseid = find_column_index(path_traza, columnID)
            column_caseid_values = extract_column(path_traza, index_caseid)
        else:
            raise ValueError(f"La columna '{columnID}' no se encuentra en el archivo.")

        # identifico cuantos y cuales casos hay en el archivo cargado de trazas
        casos, indices = np.unique(column_caseid_values,return_index=True)
        casos = casos[np.argsort(indices)]
        current_caseid=column_caseid_values[0]
        ini = 0
        fin = 0
        token_x=np.array([])
        ult_pos=len(column_caseid_values) # 6
        for c in column_caseid_values:
            #ultima_pos = 18
            if (c == current_caseid) and (ult_pos != fin+1): # avanzo mientras sea el mismo caso
                fin+=1
            else:
                if ult_pos != (fin+1): #si llego a un caso distinto y no es el final
                    rango_caseid = column_pred_values[ini:fin] #obtengo de la columna column_pred de la traza, los valores que corresponden al current_caseid
                    ini = fin
                    fin+=1
                else:
                    rango_caseid = column_pred_values[ini:ult_pos+1]

                token_x_fila = genera_fila_token_x(max_case_length, x_word_dict, rango_caseid) # genero token_x_fila
                if token_x.size ==0:
                    token_x = np.stack(token_x_fila)
                else:
                    token_x = np.vstack([token_x, token_x_fila]) # agrego a la matriz token_x

                current_caseid=c

        token_x = np.array(token_x, dtype=np.float32)

        ruta_nombreModeloTransformer = os.path.join(rutaModelo, nombreModelo + "T.keras")
        modelo = load_model(
            ruta_nombreModeloTransformer,
            custom_objects={
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "TransformerBlock": TransformerBlock,
            },
            compile=False,
        )

        y_pred = np.argmax(modelo.predict(token_x), axis=1)  # predicción

        caseid_pred=np.array([])
        index_caseid=0
        # Se traduce salida de prediccion a valores de entrada
        for y in y_pred:
            prediccion=list(y_word_dict.keys())[list(y_word_dict.values()).index(y)]
            arreglo_aux=[[casos[index_caseid],prediccion]]
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
        
        caseid_pred= formato_salida_prediccion(caseid_pred, tipoPred)

        return caseid_pred


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

