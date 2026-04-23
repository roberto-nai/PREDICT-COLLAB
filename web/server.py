"""
Main server file for the Flask web application.
Handles routing, file uploads, and interactions with models and logs.
"""
import io
import shutil
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, abort

from flask import flash


from Logs import *

from Predicciones import *
from PrediccionesTime import *
from Explainability import ShapExplainer
from FuncionesAuxiliares import *
from processtransformer import constants
from ConfigLoader import load_config, get_processing_dir, get_staging_dir


def _fmt_shap_dt(iso_str):
    """Format ISO datetime string '2026-04-22T15:29:58.968633Z' → '2026-04-22 15:29:58'."""
    if not iso_str:
        return ''
    try:
        from datetime import datetime
        return datetime.fromisoformat(iso_str.rstrip('Z')).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        return iso_str

app = Flask(__name__)

# Load configuration from config.yml
config = load_config()
app.secret_key = config['app']['secret_key']
app.debug = config['app']['debug']

# Configure logging
log_file = config['app'].get('log_file', 'app.log')
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file)

# Create a rotating file handler (max 10MB per file, keep 5 backup files)
file_handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Get Flask's logger and add the file handler
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Also configure the root logger to capture all output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        logging.StreamHandler()  # Keep console output
    ]
)

app.logger.info(f'Application started - Logging to {log_path}')

# Create processing directory if it doesn't exist
processing_dir = get_processing_dir()
if not os.path.exists(processing_dir):
    os.makedirs(processing_dir)
    # Create .gitkeep file to track empty directory in git
    gitkeep_path = os.path.join(processing_dir, '.gitkeep')
    open(gitkeep_path, 'a').close()

# Create staging directory if it doesn't exist
staging_dir = get_staging_dir()
if not os.path.exists(staging_dir):
    os.makedirs(staging_dir)
    # Create .gitkeep file to track empty directory in git
    gitkeep_path = os.path.join(staging_dir, '.gitkeep')
    open(gitkeep_path, 'a').close()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/config')
def show_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config_content = config_file.read()
    return render_template('config.html', config_content=config_content)

@app.route('/Cargarlogs',methods=['GET', 'POST'])
def get_carga_logs():
    try:
        logs = listar_logs()
        if request.method == 'POST':
            nombreProceso = request.form.get('nombreProceso')
            descripcionLog = request.form.get('descripcionLog')

            if nombreProceso == "otro":
                nombreProceso = request.form.get('nuevoProceso')
                existe_proceso = any(log['nombreProceso'].lower() == nombreProceso.lower() for log in logs)
                if existe_proceso:
                    flash('Process name already exists', 'error')
                    error = "Process " + nombreProceso + " already exists"
                    return redirect(url_for('get_logs', status="ERROR", mensaje_error=error))

                if not entrada_valida(nombreProceso):
                    flash('Process name does not meet the accepted format', 'error')
                    error = "Process name " + nombreProceso + " does not meet the required format [a-zA-Z0-9-_]"
                    return redirect(url_for('get_logs', status="ERROR",mensaje_error=error))

            file = request.files['logCsv']

            if file and file.filename.endswith(('.csv', '.xes')):
                nameLog = file.filename.rsplit('.', 1)[0]
                existe_proceso_y_log = any(
                    log['nombreProceso'].lower() == nombreProceso.lower() and log['log'].lower() == file.filename.lower()
                    for log in logs)
                if existe_proceso_y_log:
                    flash('Log name already exists in process', 'error')
                    error = "The log " + file.filename + " already exists in the selected process " + nombreProceso
                    return redirect(url_for('get_logs', status="ERROR", mensaje_error=error))

                if not entrada_valida(nameLog):
                    flash('Log name does not meet the accepted format', 'error')
                    error = "The log name " + nameLog + " does not meet the required format [a-zA-Z0-9-_]"
                    return redirect(url_for('get_logs', status="ERROR",mensaje_error=error))

                l=Logs()
                salida,extension = l.cargar_log(nombreProceso,file.filename,descripcionLog, file)
                flash('Log has been successfully saved', 'success')
                return redirect(url_for('get_logs',status="OK", salida=salida, extension=extension))
            else:
                flash('Invalid file extension', 'error')
                error = "The file " + file.filename + " is not a .csv or .xes file"
                return redirect(url_for('get_logs', status="ERROR", mensaje_error=error))


        folder_path_logs = get_processing_dir()
        procesos = filter_system_files(os.listdir(folder_path_logs))

        return redirect(url_for('get_logs', procesos=procesos,logs=logs,
                                status="OK"))
    except ValueError as e:
        flash(f'Error: {e}')
        return redirect(url_for('get_logs', status="ERROR", mensaje_error=str(e)))

    except Exception as e:
        flash(f'Unexpected error: {e}')
        return redirect(url_for('get_logs', status="ERROR", mensaje_error=str(e)))


@app.route('/descarga_log', methods=['GET']) # download log file
def descargar():
    try:
        log = request.args.get('log')
        proceso= request.args.get('proceso')

        ruta_procesos = get_processing_dir()
        ruta_proceso = os.path.join(ruta_procesos, proceso)
        ruta_carpeta_log = os.path.join(ruta_proceso, log)
        ruta_log = os.path.join(ruta_carpeta_log, log)

        return send_file(ruta_log, as_attachment=True)

    except ValueError as e:
        flash(f'Error: {e}', 'error')
        return redirect(url_for('get_logs', status="ERROR", mensaje_error=str(e)))

    except Exception as e:
        flash(f'Unexpected error: {e}')
        return redirect(url_for('get_logs', status="ERROR", mensaje_error=str(e)))

@app.route('/logs', methods=['GET', 'POST'])
def get_logs():

    if request.method == 'POST':
        log_seleccionado = request.form.get('log')
        proceso_seleccionado = request.form.get('proceso')
        return redirect(url_for('get_modelos', opcion_log=log_seleccionado, opcion_proceso=proceso_seleccionado))

    logs = listar_logs()

    folder_path_proceso = get_processing_dir()
    procesos = filter_system_files(os.listdir(folder_path_proceso))

    csv_settings = config.get('csv_settings', {})
    csv_delimiter = csv_settings.get('delimiter', ',')
    return render_template("logs.html", procesos=procesos, logs=logs, csv_delimiter=csv_delimiter)


@app.route('/obtener_logs')
def obtener_logs():

    proceso = request.args.get('proceso')
    folder_path_proceso = os.path.join(get_processing_dir(), proceso)
    logs = filter_system_files(os.listdir(folder_path_proceso))
    return jsonify({'logs': logs})

@app.route('/obtener_tiposPrediccion')
def obtener_tiposPrediccion():

    proceso = request.args.get('proceso')
    log=request.args.get('log')

    folder_path_log = os.path.join(get_processing_dir(), proceso, log)

    tiposPred=[]
    for tipoPred in filter_system_files(os.listdir(folder_path_log)):
        tipoPred_path = os.path.join(folder_path_log, tipoPred)
        if os.path.isdir(tipoPred_path):
            tiposPred.append(tipoPred)

    return jsonify({'tiposPred': tiposPred})

@app.route('/eliminar_log', methods=['GET', 'POST']) # delete log file and its associated models
def eliminar_log():
    try:
        proceso = request.args.get('proceso')
        log = request.args.get('log')

        folder_path_proceso = os.path.join(get_processing_dir(), proceso)
        folder_path_log = os.path.join(folder_path_proceso,log)

        if not existen_modelos(proceso, log):
            if os.path.exists(folder_path_log):
                shutil.rmtree(folder_path_log)
                if not os.listdir(folder_path_proceso):  # Si el directorio está vacío lo borra
                    shutil.rmtree(folder_path_proceso)

                flash('se elimino el log', 'success')
                return redirect(url_for('get_logs', status="OK", salida=log, eliminar=True))
            else:
                raise ValueError(f"Log '{folder_path_log}' does not exist.")
        else:
            raise ValueError(f"Log '{log}' has associated models, please delete them first")

    except ValueError as e:
        flash(f'Error: {e}','error')
        return redirect(url_for('get_logs', status="ERROR", mensaje_error=str(e), eliminar=False))

    except Exception as e:
        flash(f'Unexpected error: {e}', 'error')
        return redirect(url_for('get_logs', status="ERROR", mensaje_error=str(e), eliminar=False))

@app.route('/eliminar_modelo', methods=['GET', 'POST'])
def eliminar_modelo():
    try:
        proceso = request.form.get('proceso')
        log = request.form.get('log')
        tipo_prediccion = request.form.get('tipo_prediccion')
        modelo = request.form.get('modelo')

        folder_path_tipo_prediccion = os.path.join(get_processing_dir(), proceso, log, tipo_prediccion)
        folder_path_modelo = os.path.join(folder_path_tipo_prediccion,modelo)

        if os.path.exists(folder_path_modelo):
            # Borrar el directorio y su contenido
            shutil.rmtree(folder_path_modelo)

            if not os.listdir(folder_path_tipo_prediccion):  # Si el directorio está vacío
                shutil.rmtree(folder_path_tipo_prediccion)  # Borrar el directorio del proceso

            flash('Model was deleted', 'success')
            return redirect(url_for('get_modelos', status="OK", salida=modelo, eliminar=True))
        else:
            raise ValueError(f"Model '{folder_path_modelo}' does not exist.")

    except ValueError as e:
        flash(f'Error: {e}')
        return redirect(url_for('get_modelos', status="ERROR", mensaje_error=str(e),eliminar=False))

    except Exception as e:
        flash(f'Unexpected error: {e}')
        return redirect(url_for('get_modelos', status="ERROR", mensaje_error=str(e),eliminar=False))


@app.route('/obtener_modelos')
def obtener_modelos():

    proceso = request.args.get('proceso')
    log=request.args.get('log')
    tipoPred = request.args.get('tipoPred')

    folder_path_log = os.path.join(get_processing_dir(), proceso, log, tipoPred)

    modelos=[]
    for modelo in filter_system_files(os.listdir(folder_path_log)):
        modelo_path = os.path.join(folder_path_log, modelo)
        if os.path.isdir(modelo_path):
            modelos.append(modelo)

    return jsonify({'modelos': modelos})

@app.route('/obtener_participantes')
def obtener_participantes():
    try:
        proceso = request.args.get('proceso')
        log = request.args.get('log')
        column1 = request.args.get('column1')

        file_path_log = os.path.join(get_processing_dir(), proceso, log, log)

        prediction_settings = config.get('prediction', {})
        limit = prediction_settings.get('participant_fetch_limit', 200)  # max unique participants to return

        # Leer el archivo CSV en fragmentos para no cargar todo en memoria de una vez
        chunk_size = 40  # Puedes ajustar el tamaño del fragmento según tu caso
        valores_participante = set()

        for chunk in pd.read_csv(file_path_log, usecols=[column1], chunksize=chunk_size):
            # Obtener los valores únicos del fragmento
            values = chunk[column1].dropna().unique()
            # Añadir valores al conjunto de valores únicos
            valores_participante.update(values)
            # Verificar si se ha alcanzado el límite
            if len(valores_participante) > limit:
                raise ValueError(f"The number of participants in column '{column1}' exceeds {limit}.")

        participantes= list(valores_participante)

        return jsonify({'participantes': participantes})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        # Retorna un JSON genérico en caso de otro error
        return jsonify({'Unexpected error' : str(e)}), 500

@app.route('/obtener_columnas')
def obtener_columnas():
    proceso = request.args.get('proceso')
    log = request.args.get('log')
    columnas = obtener_columnas_por_proceso_y_log(proceso, log)
    return jsonify({'columnas': columnas})

@app.route('/obtener_estadisticas_log')
def obtener_estadisticas_log():
    try:
        proceso = request.args.get('proceso')
        log = request.args.get('log')
        
        file_path_log = os.path.join(get_processing_dir(), proceso, log, log)
        
        # Get column names from config
        event_log_defaults = config.get('event_log', {})
        caseid_col = event_log_defaults.get('caseid_col', 'case:concept:name')
        activity_col = event_log_defaults.get('activity_col', 'concept:name')
        timestamp_col = event_log_defaults.get('timestamp_col', 'time:timestamp')
        
        # Read the CSV file
        df = pd.read_csv(file_path_log)
        
        # Calculate statistics
        # 1. Number of distinct cases
        num_cases = df[caseid_col].nunique() if caseid_col in df.columns else 0
        
        # 2. Distinct activities with their frequencies
        activities = {}
        if activity_col in df.columns:
            activity_counts = df[activity_col].value_counts().to_dict()
            activities = {str(k): int(v) for k, v in activity_counts.items()}
        
        # 3. Column statistics (excluding caseid_col and timestamp_col)
        columns_stats = {}
        exclude_cols = [caseid_col, timestamp_col]
        for col in df.columns:
            if col not in exclude_cols:
                unique_values = df[col].dropna().unique()
                # Limit to reasonable number of values to avoid performance issues
                if len(unique_values) <= 50:
                    columns_stats[col] = [str(v) for v in unique_values]
                else:
                    columns_stats[col] = f"Too many unique values ({len(unique_values)})"
        
        return jsonify({
            'num_cases': num_cases,
            'activities': activities,
            'columns_stats': columns_stats
        })
    
    except Exception as e:
        app.logger.error(f'Error calculating log statistics: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/creaModelo')
def get_crea_modelos():
    try:
        nombreModelo = request.args.get('nombreModelo')
        proceso = request.args.get('proceso')
        log = request.args.get('log')
        column1 = request.args.get('column1')
        column2 = request.args.get('column2')
        column3 = request.args.get('column3')
        columnID = request.args.get('columnID')
        columnT = request.args.get('columnT')
        tipoPred = request.args.get('tipoPred')
        participante = request.args.get('participante')
        formatoT = request.args.get('FormatoT')


        if tipoPred in ("NextActivity", "Participant", "ActivityParticipant") :
            inputTask = Task.NEXT_ACTIVITY
        elif tipoPred in ("ParticipantSend","ActivityParticipantSend"):
            inputTask = Task.NEXT_MESSAGE_SEND
        elif tipoPred == "NextTime":
            inputTask=Task.NEXT_TIME
            formatoT = int(formatoT)
        elif tipoPred == "NextTimeMessage":
            inputTask=Task.NEXT_TIME_MESSAGE
            formatoT = int(formatoT)
        elif tipoPred == "RemainingTime":
            inputTask=Task.REMAINING_TIME
            formatoT = int(formatoT)
        elif tipoPred == "RemainingTimeParticipant":
            inputTask=Task.REMAINING_TIME_PARTICIPANT
            formatoT = int(formatoT)

        m = Modelos()
        modelos=m.listar_modelos()

        existe_modelo_en_proceso_log = any(modelo['proceso'].lower() == proceso.lower() and
                                           modelo['log'].lower() == log.lower() and
                                           modelo['tipo_prediccion'].lower() == tipoPred.lower() and
                                           modelo['modelo'].lower() == nombreModelo.lower()
                                           for modelo in modelos)
        if existe_modelo_en_proceso_log:
            flash('Existe nombre proceso', 'error')
            error = "Model " + nombreModelo+ " already exists for given process, log, prediction type"
            return redirect(url_for('get_modelos', status="ERROR", mensaje_error=error))

        if not entrada_valida(nombreModelo):
            flash('error nombre modelo', 'error')
            error = "El nombre de modelo " + nombreModelo + " no cumple el formato aceptado [a-zA-Z0-9-_]"
            return redirect(url_for('get_modelos', status="ERROR", mensaje_error=error))


        salida= m.prepare_model(proceso,log, columnID,column1,column2,column3,columnT, inputTask, tipoPred, participante,formatoT,nombreModelo)

        flash('The model has been created successfully.', 'success') # Translated message
        return redirect(url_for('get_modelos',title="Modelo_creado:",status="OK",salida=salida))

    except ValueError as e:
        flash(f'Error: {e}')
        return redirect(url_for('get_modelos', title="Modelo_creado:",status="ERROR", mensaje_error=str(e)))

    except Exception as e:
        flash(f'Error inesperado: {e}')
        return redirect(url_for('get_modelos', title="Modelo_creado:", status="ERROR", mensaje_error=str(e)))


@app.route('/models', methods=['GET', 'POST'])
def get_modelos():
    #if hay argumento modelo es porque se c reo uno, hay que mostrar ese mensaje

    if request.method == 'POST':
        modelo_seleccionado = request.form.get('modelo')
        proceso_seleccionado = request.form.get('proceso')
        log_seleccionado=request.form.get('log')
        tipo_pred_seleccionado= request.form.get('tipo_prediccion')
        return redirect(url_for('pred', opcion_modelo=modelo_seleccionado,opcion_proceso=proceso_seleccionado,opcion_log=log_seleccionado,opcion_tipopred=tipo_pred_seleccionado))

    m = Modelos()
    models = m.listar_modelos()
    models.sort(key=lambda x: x['modelo'])

    log_seleccionado = request.args.get('opcion_log')
    proceso_seleccionado = request.args.get('opcion_proceso')

    folder_path_logs = get_processing_dir()
    procesos = filter_system_files(os.listdir(folder_path_logs))

    # Get default column names from config to send it to template
    event_log_defaults = config.get('event_log', {})
    default_caseid = event_log_defaults.get('caseid_col', '')
    default_activity = event_log_defaults.get('activity_col', '')
    default_timestamp = event_log_defaults.get('timestamp_col', '')
    default_participant = event_log_defaults.get('participant_col', '')
    default_element_type = event_log_defaults.get('element_type_col', '')
    
    # Get prediction settings from config
    prediction_settings = config.get('prediction', {})
    max_decimal_places = prediction_settings.get('max_decimal_places', 4)

    return render_template('modelos.html', procesos=procesos, log_seleccionado=log_seleccionado,
                          proceso_seleccionado=proceso_seleccionado, models=models,
                          default_caseid=default_caseid, default_activity=default_activity,
                          default_timestamp=default_timestamp, default_participant=default_participant,
                          default_element_type=default_element_type, max_decimal_places=max_decimal_places)

@app.route('/predictions', methods=['GET', 'POST'])
def pred():
    try:
        if request.method == 'POST':

            file = request.files['trazaCsv']
            modelo = request.form.get('model')
            proceso = request.form.get('proceso')
            log = request.form.get('log')
            tipoPred = request.form.get('tipoPred')

            if file and file.filename.endswith(('.csv', '.xes')):

                filename = file.filename

                ruta_trazas = get_staging_dir()

                if not os.path.exists(ruta_trazas):
                    os.mkdir(ruta_trazas)

                if file.filename.endswith('.xes'):
                    extension = "xes"

                    # Se guarda traza.xes
                    file_path = os.path.join(ruta_trazas, filename)
                    file.save(file_path)

                    # convierte a DataFrame
                    traza = xes_importer.apply(file_path)
                    df = xes_converter.apply(traza, variant=xes_converter.Variants.TO_DATA_FRAME)

                    df['time:timestamp'] = df['time:timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-7]

                    nameTraza = file.filename.rsplit('.', 1)[0]
                    # Definir la ruta donde se guardará el archivo CSV
                    csv_file_path = os.path.join(ruta_trazas, nameTraza + ".csv")

                    # Guardar el DataFrame en un archivo CSV
                    df.to_csv(csv_file_path, index=False)
                    filename = nameTraza + ".csv"
                else:
                    file_path = os.path.join(ruta_trazas, filename)
                    file.save(file_path)

            if tipoPred in ("NextActivity", "Participant", "ActivityParticipant"):
                inputTask = constants.Task.NEXT_ACTIVITY
            elif tipoPred in ("ParticipantSend","ActivityParticipantSend"):
                inputTask = constants.Task.NEXT_MESSAGE_SEND
            elif tipoPred == "NextTime":
                inputTask = constants.Task.NEXT_TIME
            elif tipoPred == "NextTimeMessage":
                inputTask = constants.Task.NEXT_TIME_MESSAGE
            elif tipoPred == "RemainingTime":
                inputTask = constants.Task.REMAINING_TIME
            elif tipoPred == "RemainingTimeParticipant":
                inputTask = constants.Task.REMAINING_TIME_PARTICIPANT


            if inputTask in(constants.Task.NEXT_ACTIVITY,constants.Task.NEXT_MESSAGE_SEND):
                p = Predicciones()
                preds = p.prepare_predict(proceso, log, modelo, tipoPred, filename)
                participant = None
            else:
                p = PrediccionesNextTime()
                preds,participant = p.prepare_predict_next_time(proceso,log,modelo,tipoPred, filename)


            return render_template("resultadoprediccion.html", preds=preds, tipoPred=tipoPred, 
                                 participant=participant, proceso=proceso, log=log, filename=filename)

        folder_path_logs = get_processing_dir()
        # Processing directory is created at application startup
        # if not os.path.exists(folder_path_logs):
        #     os.mkdir(folder_path_logs)

        procesos = filter_system_files(os.listdir(folder_path_logs))

        log_seleccionado = request.args.get('opcion_log')
        proceso_seleccionado = request.args.get('opcion_proceso')
        modelo_seleccionado = request.args.get('opcion_modelo')
        tipo_prediccion_seleccionado=request.args.get('opcion_tipopred')

        return render_template("predicciones.html", procesos=procesos, log_seleccionado=log_seleccionado, proceso_seleccionado=proceso_seleccionado,
                                modelo_seleccionado=modelo_seleccionado,tipo_prediccion_seleccionado=tipo_prediccion_seleccionado )

    except ValueError as e:
        flash(f'Error: {e}')
        return redirect(url_for('pred', title="Modelo_creado:",status="ERROR", mensaje_error=str(e)))

    except Exception as e:
        flash(f'Unexpected error: {e}')
        return redirect(url_for('pred', title="Modelo_creado:", status="ERROR", mensaje_error=str(e)))

@app.route('/obtener_detalle_caso')
def obtener_detalle_caso():
    try:
        proceso = request.args.get('proceso')
        log = request.args.get('log')
        filename = request.args.get('filename')
        case_id = request.args.get('case_id')
        
        app.logger.info(f'Retrieving case details for case_id: {case_id}, proceso: {proceso}, log: {log}, filename: {filename}')
        
        # Get column names from config
        event_log_defaults = config.get('event_log', {})
        caseid_col = event_log_defaults.get('caseid_col', 'case:concept:name')
        timestamp_col = event_log_defaults.get('timestamp_col', 'time:timestamp')
        activity_col = event_log_defaults.get('activity_col', 'concept:name')
        
        # Read complete event log
        file_path_log = os.path.join(get_processing_dir(), proceso, log, log)
        df_complete = pd.read_csv(file_path_log)
        
        app.logger.info(f'Complete log columns: {df_complete.columns.tolist()}')
        app.logger.info(f'Complete log case_id column ({caseid_col}) dtype: {df_complete[caseid_col].dtype if caseid_col in df_complete.columns else "NOT FOUND"}')
        app.logger.info(f'Received case_id: {case_id} (type: {type(case_id).__name__})')
        
        # Try to convert case_id to match the column type
        if caseid_col in df_complete.columns:
            # Get unique case IDs for debugging
            unique_case_ids = df_complete[caseid_col].unique()
            app.logger.info(f'Sample of unique case IDs in complete log: {unique_case_ids[:5].tolist() if len(unique_case_ids) > 0 else []}')
            
            # Try to match the type of the case_id column
            try:
                # If the column is numeric, convert case_id to numeric
                if pd.api.types.is_numeric_dtype(df_complete[caseid_col]):
                    case_id_converted = pd.to_numeric(case_id)
                    app.logger.info(f'Converted case_id to numeric: {case_id_converted}')
                else:
                    case_id_converted = str(case_id)
                    app.logger.info(f'Converted case_id to string: {case_id_converted}')
            except Exception as e:
                app.logger.error(f'Error converting case_id: {e}')
                case_id_converted = case_id
        else:
            app.logger.error(f'Case ID column {caseid_col} not found in complete log!')
            case_id_converted = case_id
        
        # Filter by case_id and sort by timestamp
        df_case_complete = df_complete[df_complete[caseid_col] == case_id_converted].copy()
        
        app.logger.info(f'Number of rows found in complete log for case_id {case_id_converted}: {len(df_case_complete)}')
        
        if len(df_case_complete) == 0:
            app.logger.warning(f'No data found for case_id: {case_id_converted} in complete log')
            # Try alternative matching strategies
            app.logger.info('Attempting alternative matching strategies...')
            # Try string comparison
            df_case_complete = df_complete[df_complete[caseid_col].astype(str) == str(case_id)].copy()
            if len(df_case_complete) > 0:
                app.logger.info(f'Found {len(df_case_complete)} rows using string comparison')
            else:
                # Try stripping whitespace
                df_case_complete = df_complete[df_complete[caseid_col].astype(str).str.strip() == str(case_id).strip()].copy()
                if len(df_case_complete) > 0:
                    app.logger.info(f'Found {len(df_case_complete)} rows using string comparison with strip()')
        
        df_case_complete = df_case_complete.sort_values(by=timestamp_col)
        
        # Read trace file (partial activities)
        ruta_trazas = get_staging_dir()
        file_path_trace = os.path.join(ruta_trazas, filename)
        
        df_trace = pd.read_csv(file_path_trace)
        
        # Try to convert case_id to match the column type in trace file
        if caseid_col in df_trace.columns:
            try:
                if pd.api.types.is_numeric_dtype(df_trace[caseid_col]):
                    case_id_trace_converted = pd.to_numeric(case_id)
                else:
                    case_id_trace_converted = str(case_id)
            except:
                case_id_trace_converted = case_id
        else:
            case_id_trace_converted = case_id
        
        # Filter by case_id and sort by timestamp
        df_case_trace = df_trace[df_trace[caseid_col] == case_id_trace_converted].copy()
        
        if len(df_case_trace) == 0:
            app.logger.warning(f'No data found for case_id: {case_id_trace_converted} in trace file')
        
        df_case_trace = df_case_trace.sort_values(by=timestamp_col)
        
        # Reorder columns: caseid, timestamp, activity, then common columns, then unique columns
        def reorder_columns(df1, df2, caseid_col, timestamp_col, activity_col):
            """Reorder columns so both dataframes have the same column order"""
            # Get all columns from both dataframes
            cols1 = set(df1.columns)
            cols2 = set(df2.columns)
            
            # Find common and unique columns
            common_cols = cols1.intersection(cols2)
            unique_cols1 = cols1 - cols2
            unique_cols2 = cols2 - cols1
            
            # Start with the priority columns (caseid, timestamp, activity)
            ordered_cols = []
            if caseid_col in common_cols:
                ordered_cols.append(caseid_col)
                common_cols.remove(caseid_col)
            if timestamp_col in common_cols:
                ordered_cols.append(timestamp_col)
                common_cols.remove(timestamp_col)
            if activity_col in common_cols:
                ordered_cols.append(activity_col)
                common_cols.remove(activity_col)
            
            # Add remaining common columns (sorted alphabetically)
            ordered_cols.extend(sorted(common_cols))
            
            # For df1: add its unique columns at the end
            df1_cols = ordered_cols + sorted(unique_cols1)
            
            # For df2: add its unique columns at the end
            df2_cols = ordered_cols + sorted(unique_cols2)
            
            return df1_cols, df2_cols
        
        # Reorder columns for both dataframes
        complete_cols_ordered, trace_cols_ordered = reorder_columns(
            df_case_complete, df_case_trace, caseid_col, timestamp_col, activity_col
        )
        
        # Apply column reordering
        df_case_complete = df_case_complete[complete_cols_ordered]
        df_case_trace = df_case_trace[trace_cols_ordered]
        
        # Convert to list of dictionaries and replace NaN/NaT with None
        df_case_complete = df_case_complete.fillna('')
        df_case_trace = df_case_trace.fillna('')
        complete_case_data = df_case_complete.to_dict('records')
        trace_case_data = df_case_trace.to_dict('records')
        complete_case_columns = df_case_complete.columns.tolist()
        trace_case_columns = df_case_trace.columns.tolist()
        
        return jsonify({
            'complete_case': {
                'columns': complete_case_columns,
                'data': complete_case_data
            },
            'trace_case': {
                'columns': trace_case_columns,
                'data': trace_case_data
            }
        })
    
    except Exception as e:
        app.logger.error(f'Error retrieving case details: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download_csv', methods=['POST'])
def download_csv():
    json_data = request.get_json()
    data = json_data['data']
    prediction_header = json_data['predictionHeader']

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Case', prediction_header])

    # Escribe los datos de la tabla
    for row in data:
        writer.writerow(row)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='Prediction_results.csv'
    )

@app.route('/explain_model', methods=['POST'])
def explain_model():
    try:
        proceso = request.form.get('proceso')
        log = request.form.get('log')
        tipo_prediccion = request.form.get('tipo_prediccion')
        modelo = request.form.get('modelo')

        if not all([proceso, log, tipo_prediccion, modelo]):
            raise ValueError('Missing required model parameters for SHAP explainability.')

        explainer = ShapExplainer()
        result = explainer.explain_model(
            process_name=proceso,
            log_name=log,
            prediction_type=tipo_prediccion,
            model_name=modelo,
            top_k=3,
            nsamples=100,
        )

        return render_template(
            'shap_results.html',
            proceso=proceso,
            log=log,
            tipo_prediccion=tipo_prediccion,
            modelo=modelo,
            explained_cases=result['explained_cases'],
            shap_rows=result['rows'],
            shap_summary_rows=result['summary_rows'],
            output_dir=result['output_dir'],
            detail_csv_path=result['detail_csv_path'],
            summary_csv_path=result['summary_csv_path'],
            metadata_path=result['metadata_path'],
            cached=result.get('cached', False),
            started_at=_fmt_shap_dt(result.get('started_at', '')),
            ended_at=_fmt_shap_dt(result.get('ended_at', '')),
            delta_time_sec=result.get('delta_time_sec', ''),
            delta_time_min=result.get('delta_time_min', ''),
        )

    except ValueError as e:
        flash(f'Error: {e}', 'error')
        return redirect(url_for('get_modelos', status='ERROR', mensaje_error=str(e)))

    except Exception as e:
        app.logger.error(f'Unexpected SHAP explain error: {e}', exc_info=True)
        flash(f'Unexpected error: {e}', 'error')
        return redirect(url_for('get_modelos', status='ERROR', mensaje_error=str(e)))


@app.route('/faq')
def info():
    return render_template('faq.html')


if __name__ == "__main__":
    print()
    print("*** PROGRAM START ***")
    print()
    print("Config loaded:")
    print(config)
    print()

    app.run(
        debug=config['app']['debug'],
        port=config['app']['port'],
        host=config['app']['host']
    )
    #serve(app, host="0.0.0.0", port=8000)

    print()
    print("*** PROGRAM END ***")
    print()

