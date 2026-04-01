"""
Docstring per web.Logs
"""
import joblib
import tensorflow as tf
import os
import numpy as np

from FuncionesAuxiliares import *
from ConfigLoader import get_processing_dir

from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer


class Logs:

    def cargar_log(self,nombreProceso,filename,descripcionLog, file):

        nameLog = file.filename.rsplit('.', 1)[0]

        ruta_procesos = get_processing_dir()
        ruta_nombre_proceso = os.path.join(ruta_procesos, nombreProceso)
        ruta_name_log= os.path.join(ruta_nombre_proceso,nameLog + ".csv")


        # si no existe el proceso, se crea carpeta
        if not os.path.exists(ruta_nombre_proceso):
            os.mkdir(ruta_nombre_proceso)

        # si no existe el log , crea la carpeta
        if not os.path.exists(ruta_name_log):
            os.mkdir(ruta_name_log)

        if file.filename.endswith('.xes'):
            extension="xes"

            #Se guarda xes
            file_path = os.path.join(ruta_name_log, filename)
            file.save(file_path)

            #convierte a DataFrame
            log = xes_importer.apply(file_path)
            df = xes_converter.apply(log, variant=xes_converter.Variants.TO_DATA_FRAME)

            df['time:timestamp'] = df['time:timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-7]

            # Definir la ruta donde se guardará el archivo CSV
            csv_file_path = os.path.join(ruta_name_log, nameLog + ".csv")

            # Guardar el DataFrame en un archivo CSV
            df.to_csv(csv_file_path, index=False)

            file_path=csv_file_path
        else:
            extension = "csv"
            file_path = os.path.join(ruta_name_log, filename)
            file.save(file_path)

        nombres_columnas = obtener_nombre_columnas(file_path)

        if nombres_columnas == []:
            return 'Fila 1 vacia'

        save_variables(os.path.join(ruta_name_log,"propiedadesLog.txt"), descripcionLog, nombres_columnas, nombreProceso)

        return filename,extension