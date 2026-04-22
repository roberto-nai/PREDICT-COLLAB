"""
Docstring per web.Modelos

"""
import joblib
import tensorflow as tf
import shutil

from sklearn import metrics

from processtransformer.constants import Task

from processtransformer.models import transformer
from processtransformer.data.loader import LogsDataLoader
from processtransformer.data.processor import LogsDataProcessor
from FuncionesAuxiliares import *
from ConfigLoader import get_processing_dir

class Modelos:
    def __init__(self):
        self.atributo1 = "sdf"

    def prepare_model(self,proceso, log, columnID, column1, column2, column3,columnT, inputTask, tipoPred, participant,formatoT, nombreModelo):

        ruta_procesos = get_processing_dir()
        ruta_proceso=os.path.join(ruta_procesos, proceso)
        ruta_carpeta_log=os.path.join(ruta_proceso,log)
        ruta_log=os.path.join(ruta_carpeta_log, log)
        ruta_log_temporal = None
        ruta_log_procesamiento = ruta_log


        columnaFinal=""
        if tipoPred in ("NextActivity", "Participant","NextTime","NextTimeMessage","RemainingTime","RemainingTimeParticipant"):
            columnaFinal = column1
        elif tipoPred in ("ParticipantSend","ActivityParticipant"):
            ruta_log_temporal = os.path.join(ruta_carpeta_log, f".__tmp_model_input__{log}")
            shutil.copyfile(ruta_log, ruta_log_temporal)
            ruta_log_procesamiento = ruta_log_temporal
            columnaFinal = concatenate_columns(ruta_log_procesamiento, column1, column2, None)
        elif tipoPred == "ActivityParticipantSend":
            ruta_log_temporal = os.path.join(ruta_carpeta_log, f".__tmp_model_input__{log}")
            shutil.copyfile(ruta_log, ruta_log_temporal)
            ruta_log_procesamiento = ruta_log_temporal
            columnaFinal = concatenate_columns(ruta_log_procesamiento, column1, column2, column3)

        # dejo en minuscula  participant
        if participant is not None:
            participant=participant.lower()

        ruta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_base = os.path.abspath(os.path.join(ruta_actual, os.pardir))
        ruta_datasets = os.path.join(ruta_base,'web', 'datasets')

        data_processor = LogsDataProcessor(name='predict-collab', filepath=ruta_log_procesamiento,
                                           columns=[columnID, columnaFinal, columnT],
                                           dir_path=ruta_datasets, pool=4)

        # Arma los csv para train y test
        data_processor.process_logs(task=inputTask, participant=participant,formatoT=formatoT,sort_temporally=False)

       # Load data
        data_loader = LogsDataLoader(name='predict-collab', dir_path=ruta_datasets)

        (train_df, test_df, x_word_dict, y_word_dict, max_case_length,
         vocab_size, num_output) = data_loader.load_data(inputTask, participant)


        learning_rate = 0.001
        batch_size = 12
        epochs = 10

        # Segun que tipo de prediccion es, hago el prepare data correspondiente y el entrenamiento
        if inputTask in (Task.NEXT_ACTIVITY,Task.NEXT_MESSAGE_SEND):
            train_token_x, train_token_y = data_loader.prepare_data_next_activity(train_df,x_word_dict, y_word_dict, max_case_length)
            # Create and train a transformer model NEXT ACTIVITY
            transformer_model = transformer.get_next_activity_model(
                max_case_length=max_case_length,
                vocab_size=vocab_size,
                output_dim=num_output)

            transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

            transformer_model.fit(train_token_x, train_token_y,
                                  epochs=epochs, batch_size=batch_size)

            k, accuracies, fscores, precisions, recalls = [], [], [], [], []
            for i in range(max_case_length):
                test_data_subset = test_df[test_df["k"] == i]
                if len(test_data_subset) > 0:
                    test_token_x, test_token_y = data_loader.prepare_data_next_activity(test_data_subset,
                                                                                        x_word_dict, y_word_dict,
                                                                                        max_case_length)
                    y_pred = np.argmax(transformer_model.predict(test_token_x), axis=1)
                    accuracy = metrics.accuracy_score(test_token_y, y_pred)
                    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                        test_token_y, y_pred, average="weighted")
                    k.append(i)
                    accuracies.append(accuracy)
                    fscores.append(fscore)
                    precisions.append(precision)
                    recalls.append(recall)

            k.append(i + 1)
            accuracies.append(np.mean(accuracy))
            fscores.append(np.mean(fscores))
            precisions.append(np.mean(precisions))
            recalls.append(np.mean(recalls))

        elif inputTask in (Task.NEXT_TIME,Task.NEXT_TIME_MESSAGE):
            (train_token_x, train_time_x, train_token_y, time_scaler, y_scaler) = data_loader.prepare_data_next_time(train_df, x_word_dict, max_case_length)

            # Create and train a transformer model NEXT TIME
            transformer_model = transformer.get_next_time_model(
                max_case_length=max_case_length,
                vocab_size=vocab_size)

            transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                                      loss=tf.keras.losses.LogCosh())

            transformer_model.fit([train_token_x, train_time_x], train_token_y,
                                  epochs=epochs, batch_size=batch_size, verbose=2)



            # --------Generar Metricas Segun Next time -------------------
            k, maes, mses, rmses = [], [], [], []
            for i in range(max_case_length):
                test_data_subset = test_df[test_df["k"] == i]
                if len(test_data_subset) > 0:
                    test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_next_time(
                        test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False)

                    y_pred = transformer_model.predict([test_token_x, test_time_x])
                    _test_y = y_scaler.inverse_transform(test_y)
                    _y_pred = y_scaler.inverse_transform(y_pred)


                    k.append(i)
                    maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
                    mses.append(metrics.mean_squared_error(_test_y, _y_pred))
                    rmses.append(np.sqrt(metrics.mean_squared_error(_test_y, _y_pred)))

            k.append(i + 1)
            maes.append(np.mean(maes))
            mses.append(np.mean(mses))
            rmses.append(np.mean(rmses))
            # --------------------------------------------

        elif inputTask in (Task.REMAINING_TIME, Task.REMAINING_TIME_PARTICIPANT):
            (train_token_x, train_time_x, train_token_y, time_scaler, y_scaler) = data_loader.prepare_data_remaining_time(train_df, x_word_dict, max_case_length)

            # Create and train a transformer model REMAINING TIME
            transformer_model = transformer.get_remaining_time_model(
                max_case_length=max_case_length,
                vocab_size=vocab_size)

            transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                                      loss=tf.keras.losses.LogCosh())

            transformer_model.fit([train_token_x, train_time_x], train_token_y,
                                  epochs=epochs, batch_size=batch_size, verbose=2)


            # --------Generar Metricas Segun Remaining time -------------------
            k, maes, mses, rmses = [], [], [], []
            for i in range(max_case_length):
                test_data_subset = test_df[test_df["k"] == i]
                if len(test_data_subset) > 0:
                    test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_remaining_time(
                        test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False)

                    y_pred = transformer_model.predict([test_token_x, test_time_x])
                    _test_y = y_scaler.inverse_transform(test_y)
                    _y_pred = y_scaler.inverse_transform(y_pred)

                    k.append(i)
                    maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
                    mses.append(metrics.mean_squared_error(_test_y, _y_pred))
                    rmses.append(np.sqrt(metrics.mean_squared_error(_test_y, _y_pred)))

            k.append(i + 1)
            maes.append(np.mean(maes))
            mses.append(np.mean(mses))
            rmses.append(np.mean(rmses))
        # ---------------

        # Crear directorios del modelo

        # si no existe el tipo de prediccion para el log , crea la carpeta para el tipoPrediccion

        ruta_tipoPred=os.path.join(ruta_carpeta_log,tipoPred)
        if not os.path.exists(ruta_tipoPred):
            os.mkdir(ruta_tipoPred)

        # creo carpeta NombreModelo
        ruta_nombreModelo=os.path.join(ruta_tipoPred,nombreModelo)
        if not os.path.exists(ruta_nombreModelo):
            os.mkdir(ruta_nombreModelo)

        #Crea carpeta con modelo transformer
        ruta_nombreModeloTransformer=os.path.join(ruta_nombreModelo,nombreModelo + "T.keras")

        transformer_model.save(ruta_nombreModeloTransformer , overwrite=True)

       #Creo el archivo de propiedades del modelo
        propiedadesModelo = os.path.join(ruta_nombreModelo,'propiedadesModelo.txt')

        # Segun que tipo de modelo,guardo las variables de metricas distintas
        if inputTask in (Task.NEXT_ACTIVITY, Task.NEXT_MESSAGE_SEND):
            metrica1=np.mean(accuracies)
            metrica2=np.mean(fscores)
            metrica3=np.mean(precisions)
            metrica4=np.mean(recalls)
            # Save variables to file
            save_variables(propiedadesModelo, x_word_dict, y_word_dict, max_case_length,columnID,columnT,column1,column2,column3,
                           inputTask,tipoPred, formatoT,participant,metrica1,metrica2,metrica3,metrica4)
            # Save metrics to CSV
            save_model_metrics_to_csv(ruta_carpeta_log, proceso, log, nombreModelo, tipoPred, metrica1, metrica2, metrica3, metrica4)
        else:
            metrica1 = np.mean(maes)
            metrica2 = np.mean(mses)
            metrica3 = np.mean(rmses)
            metrica4= None
            # Save variables to file
            save_variables(propiedadesModelo, x_word_dict, y_word_dict, max_case_length, columnID, columnT, column1,column2,column3,
                           inputTask,tipoPred, formatoT, participant, metrica1, metrica2, metrica3,metrica4)
            # Save metrics to CSV
            save_model_metrics_to_csv(ruta_carpeta_log, proceso, log, nombreModelo, tipoPred, metrica1, metrica2, metrica3, metrica4)

        # Creo archivo con y_scaler y time_scaler

        if inputTask not in (Task.NEXT_ACTIVITY,Task.NEXT_MESSAGE_SEND):
            ruta_scalers= os.path.join(ruta_nombreModelo,"scalers")
            if not os.path.exists(ruta_scalers):
                os.mkdir(ruta_scalers)

            ruta_absoluta_y_scaler = os.path.join(ruta_scalers, "y_scaler.pkl")
            joblib.dump(y_scaler, ruta_absoluta_y_scaler)

            ruta_absoluta_time_scaler = os.path.join(ruta_scalers,"time_scaler.pkl")

            joblib.dump(time_scaler, ruta_absoluta_time_scaler)

        # ---------------

        if ruta_log_temporal is not None and os.path.exists(ruta_log_temporal):
            os.remove(ruta_log_temporal)

        return nombreModelo

    def listar_modelos(self):
        ruta_procesos = get_processing_dir()

        # Processing directory is created at application startup
        # if not os.path.exists(ruta_procesos):
        #     os.mkdir(ruta_procesos)

        modelos = []
        # Recorrer la estructura de directorios
        for proceso in filter_system_files(os.listdir(ruta_procesos)):
            proceso_path = os.path.join(ruta_procesos, proceso)
            if os.path.isdir(proceso_path):
                for log in filter_system_files(os.listdir(proceso_path)):
                    log_path = os.path.join(proceso_path, log)
                    if os.path.isdir(log_path):
                        for tipo_prediccion in filter_system_files(os.listdir(log_path)):
                            tipo_prediccion_path = os.path.join(log_path, tipo_prediccion)
                            if os.path.isdir(tipo_prediccion_path):
                                for modelo in filter_system_files(os.listdir(tipo_prediccion_path)):
                                    modelo_path = os.path.join(tipo_prediccion_path, modelo)
                                    if os.path.isdir(modelo_path):
                                        prop = load_variables(os.path.join(modelo_path, "propiedadesModelo.txt"))
                                        met1 = prop[12]
                                        met2 = prop[13]
                                        met3 = prop[14]
                                        met4 = prop[15]
                                        texto_tipo_prediccion = obtener_texto_tipoPred(tipo_prediccion)
                                        modelos.append({
                                            'proceso': proceso,
                                            'log': log,
                                            'tipo_prediccion': tipo_prediccion,
                                            'texto_tipo_prediccion': texto_tipo_prediccion,
                                            'modelo': modelo,
                                            'id': modelo_path,  # Puedes usar el modelo_path o un id más específico
                                            'met1': met1,
                                            'met2': met2,
                                            'met3': met3,
                                            'met4': met4
                                        })
        return modelos


def obtener_texto_tipoPred(tipoPred):
    switch_dict = {
        "NextActivity": "Next activity that is likely to occur in the process",
        "ParticipantSend": "Next participant that is likely to send a message",
        "ActivityParticipantSend": "Next participant that is likely to send a message(with activity)",
        "ActivityParticipant": "Next activity that is likely to occur in the process and its participant",
        "Participant": "Next participant that is likely to act",
        "NextTime": "Time until the next event",
        "NextTimeMessage": "Time until the next message to send",
        "RemainingTime": "Process remaining time",
        "RemainingTimeParticipant": "Participant remaining time",
    }
    # Retorna el valor de tipoPred si existe, de lo contrario, retorna un texto vacío
    return switch_dict.get(tipoPred, "")

