# Predict-Collab: A Tool for Predictions in Collaborative Business Processes based on ProcessTransformer

> **Note:** This is a fork of the original project available at https://gitlab.fing.edu.uy/open-coal/predict-collab/.   
> See [UPDATES.txt](UPDATES.txt) for changes and improvements made to the original project.

Predictive monitoring of business processes aims to anticipate the future behavior of business processes by analyzing historical and real-time data.
This discipline has primarily focused on processes carried out within a single organization (intra-organizational processes) 
and not on collaborative processes involving two or more organizations (inter-organizational processes). 
We propose extending the ProcessTransformer tool's predictive capabilities for collaborative scenarios to address this. 
The solution was evaluated using extended event logs for collaborative processes, and the development was carried out in Python, along with the Flask microframework for web application development.

We provide collaborative predictions for:

- Next activity that is likely to occur in the process
- Next participant that is likely to send a message
- Next participant that is likely to send a message (with activity)
- Next activity that is likely to occur in the process and its participant
- Next participant that is likely to act
- Time until the next event
- Time until the next message to send
- Process remaining time
- Participant remaining time


> **Paper:** Daniel Calegari, Andrea Delgado: Extending predictive process monitoring for collaborative processes. CoRR abs/2409.09212 (2024) [LINK](https://arxiv.org/abs/2409.09212)

> **Partially supported by project** "Minería de procesos y datos para la mejora de procesos colaborativos aplicada a e-Government" funded by Agencia Nacional de Investigación e Innovación (ANII), Fondo María Viñas (FMV) "Proyecto ANII N° FMV_1_2021_1_167483", Uruguay

## Contributors

- Andrea Delgado
- Daniel Calegari
- Roberto Nai (roberto.nai@unito.it)
- Emilio Sulis


# Instructions
We provide the necessary code to use Predict-Collab with the event logs of your choice. 

### For Windows 11 or Mac (tested with Sequoia)
You need to install python 3.10 with pip then follow this steps

1. Open terminal in the app's web directory 
2. Check python version to be 3.10
	```py --version``` (use ```python3``` instead of ```py``` in macOS)
3. Create virtual enviroment
	```py -m venv venv```
4. Activate the virtual enviroment
	```venv\Scripts\activate.bat``` (use ```source ./venv/bin/activate``` in macOS)
5. Install requirements
	```pip install -r requirements.txt```
6. Enter the ```web``` directory and run the app
	```python server.py```
7. Open your navigator with the URL provided in the terminal (e.g.: "Running on http://127.0.0.1:8000" >> Open http://127.0.0.1:8000 in the browser)

**Note:** if "No space left on the device" appears when installing the required software, create a ```tmp``` directory in your web directory and run ```pip install --cache-dir=PATH_TO_WEB\tmp -r requirements.txt```

### Requirements
Install project dependencies using [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

- <a href="https://github.com/Zaharah/processtransformer">ProcessTransformer </a>


## Project Structure

> **Note:** See [UPDATES.txt](UPDATES.txt) for changes and improvements made to the original project.

The project is organized in the following directories:

1. **experimentation/** - Contains the event logs used for testing and evaluation:
   - Event logs for different scenarios (artificial1, artificial5, healthcare, real4)
   - Each scenario includes collective logs, XES files, training/test folds, and partial traces
   
2. **web/** - Contains the main application code:
   - Flask server implementation (`server.py` to be executed from ./)
   - Python modules for models, predictions, and auxiliary functions
   - ProcessTransformer integration
   - `static/` and `templates/` are Flask built-in directories for static resources and HTML templates

3. **trazas/** - Stores the files uploaded through the web application.

4. **processing/** - Stores the prediction model processing outputs for each individual event log.


## References

Zaharah A. Bukhsh, Aaqib Saeed, & Remco M. Dijkman. (2021). ["ProcessTransformer: Predictive Business Process Monitoring with Transformer Network"](https://arxiv.org/abs/2104.00721). arXiv preprint arXiv:2104.00721   

Delgado A., Calegari D., Espino C. & Ribero N. (2025).  ["Predictive process monitoring for collaborative business processes: concepts and application](https://link.springer.com/article/10.1007/s44257-025-00031-8)
