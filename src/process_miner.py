# activate venv -> .\.venv\Scripts\activate --- source .venv/bin/activate
# PATH -> export PYTHONPATH="${PYTHONPATH}:/home/ruben/Documents/TFG/" ## echo $PYTHONPATH para verlo
import optimize
import parameters

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal_fixed import NSGAIII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import time
import os
import pandas as pd

class ProcessMiner:
    '''
    A class with the objective of mining useful process models from event logs using various optimization algorithms.

    The actual optimizacion process is abstracted to the class 'optimize.Optimizer()'.
    Here the user can specify log files, the out folder for the petri net representation of the processes among others.
    
    The result of the execution is an approximation ot the pareto front of the optimal solutions returned from the optimizer.

    Attributes
    ----------
    log_file : str
        The path to the log file where results will be recorded.
    out_folder : str
        The output folder path where results and visualizations will be saved.
    miner_type : str
        The type of mining algorithm to be used (e.g., "Heuristic").
    metrics_type : str
        The string defining the type of metrics used for evaluation. Must be one of te implementations from the 'metrics' module.
    log_name : str
        The desired path to the log file, the file will be created if does not exist
    '''

    log_file = 'doc/log.csv'
    out_folder = 'out/'
    available_opts = {'NSGAII' : NSGAII,
                      'NSGAIII' : NSGAIII,
                      'SPEA2': SPEA2}


    def __init__(self, miner_type, metrics,  log, outpath=None):

        self.miner_type = miner_type
        self.metrics_type = metrics
        self.log_name = os.path.basename(log)

        self.miner = self.__get_miner_alg(miner_type)
        self.log = xes_importer.apply(log)
        self.metrics_obj = self.__get_metrics_type(metrics)
        
        self.local_time = time.strftime("[%Y_%m_%d - %H:%M:%S]", time.localtime())

        if outpath == None:
            self.outpath = f'{self.out_folder}/{self.local_time}-{self.log_name}'
        else:
            self.outpath = f'{self.out_folder}/{outpath}'

        self.opt = optimize.Optimizer(self.miner, self.log, self.metrics_obj, self.outpath)
        self.star_time = time.time()
        
    def __get_miner_alg(self, miner):
        '''
        Retrieves the mining algorithm class based on the specified miner name.

        Returns
        -------
        class : The mining algorithm class corresponding to the specified miner name.
        '''
        if miner not in parameters.miner_mapping:
            raise ValueError(f"Minero '{miner}' no está soportado. Los mineros disponibles son: {list(parameters.miner_mapping.keys())}")
        return parameters.miner_mapping[miner]

    def __get_metrics_type(self, metrics):
        '''
        Retrieves the metrics class based on the specified name.
        
        Returns
        -------
        class : The metrics class corresponding to the specified metrics name.
        '''
        if metrics not in parameters.metrics_mapping:
            raise ValueError(f"Las métricas '{metrics}' no están soportadas. Las métricas disponibles son: {list(parameters.metrics_mapping.keys())}")
        return parameters.metrics_mapping[metrics]
    
    def __log(self):
        '''
        Stores relevant information about the execution (e.g. runtime, log name, miner type, etc)
        '''
        if os.path.isfile(self.log_file):
            with open(self.log_file, 'a') as log:
                runtime = str(self.end_time - self.star_time)
                log.write(f'\n{self.local_time};{runtime};{self.log_name};{self.miner_type};{self.opt_type};{self.extract_params()};{self.metrics_type};{self.opt.get_best_solution().variables}')
        else:
            with open(self.log_file, 'w') as log:
                log.write('Timestamp,Runtime, Log Name, Miner Type, Opt type, Opt Parameters, Metrics type, Optimal solution')
            self.__log()

    def extract_params(self):
        '''
        Extracts the parameters of the optimization algorithm (e.g. NSGAII) into strings

        Returns
        -------

        dict : Dictionary where the keys are the parameter names and the values are either 
               the corresponding parameter value or, for objects, a list containing the object
               and its atributes.
        '''
        dicc = dict()
        for (attr_name, attr_value) in self.params.items():
            if hasattr(attr_value, '__dict__'):
                dicc[attr_name] = [f"{attr_value.__class__.__module__}.{attr_value.__class__.__name__}", attr_value.__dict__]
            else:
                dicc[attr_name] = attr_value
        return dicc
    
    def save_petri_nets(self):
        """
        Saves Petri nets for each Pareto optimal solution.
        """
        os.makedirs(self.outpath, exist_ok=True)

        for index, petri in enumerate(self.opt.get_pareto_front_petri_nets()):
            gviz = pn_visualizer.apply(petri[0], petri[1], petri[2])
            pn_visualizer.save(gviz, f'{self.outpath}/petri_pareto_{index}.png')

    def save_pareto_front(self):
        """
        Saves the Pareto front graph as an image.
        """
        os.makedirs(self.outpath, exist_ok=True)
        self.opt.plot_pareto_front(title='Pareto front approximation', filename=f'{self.outpath}/Pareto Front')

    def save_csvs(self):
        """
        Saves CSV files for variables and objectives of each Pareto optimal solution.
        """
        os.makedirs(self.outpath, exist_ok=True)

        # Save result variables to CSV
        with open(f"{self.outpath}/results_variables.csv", 'w') as log:
            parameter_names = ",".join(self.opt.parameters_info.base_params.keys())
            log.write(parameter_names + "\n")
            for sol in self.opt.get_result():
                log.write(f'{",".join(map(str, sol.variables))}\n')

        # Save result objectives to CSV
        with open(f"{self.outpath}/results_objectives.csv", 'w') as log:
            metrics_labels = ",".join(self.metrics_obj.get_labels())
            log.write(metrics_labels + "\n")
            for sol in self.opt.get_result():
                log.write(f'{",".join(map(str, sol.objectives))}\n')


    def discover(self, algorithm_name, store=True, **params):
        '''
        Performs the hiperparameter optimization of the miner specified when instanciating
        the class using the algorithm class and its hiperparameters entered as attributes.
        If the 'store' parameter is True relevant information of the execution will be saved

        It does not return anything, but other methods are available to retrieve information
        about the execution.

        Parameters
        ----------
        algorithm_name : class
            The name of the optimization algorithm to be used. Must be one of the avalilable names.
        **params : dict, optional
            Additional keyword arguments representing the hyperparameters to be passed to the algorithm class.
        
        '''
        if algorithm_name not in self.available_opts:
            return ValueError(f"Optmizador '{algorithm_name}' no está soportado. Los optmizadores disponibles son: {list(self.available_opts.keys())}")
        else:
            algorithm_class = self.available_opts[algorithm_name]

        self.params = params
        self.opt.discover(algorithm_class=algorithm_class, **params)
        self.opt_type = algorithm_class.__name__
        self.end_time = time.time()
        self.__log()
        if store:
            self.save_petri_nets()
            self.save_pareto_front()
            self.save_csvs()

    
    def set_log_file(self, logpath):
        '''
        Allows to set up the log file path (default is 'doc/log.csv')
        '''
        self.log_file = logpath
    
    def set_out_folder(self, outpath):
        '''
        Allows to set up the out path where information about each execution will be stored
        (default is 'out/')
        '''
        os.makedirs(outpath, exist_ok=True)
        self.out_folder = outpath

    def show_pareto_iterations(self):
        fronts_path = os.path.join(self.outpath, 'FRONTS')
        files = sorted([f for f in os.listdir(fronts_path) if f.startswith("FUN.") and os.path.isfile(os.path.join(fronts_path, f))])
        
        for file in files:
            with open(os.path.join(fronts_path, file), 'r') as f:
                lines = f.readlines()
                unique_lines = sorted(set(map(str.strip, lines)))
                unique_count = len(unique_lines)
                print(f"{file}: {unique_count}")

def generate_dataset(miner_type:str, metrics:str, logs:list[tuple[str,str]], outpath:str, iters:int, optim_params=parameters.nsgaii_params, optim_name='NSGAII'):

    for i in range(iters):
        print(f"Iteración {i+1} de {iters}")
        for log_name, log in logs:

            path = f'{outpath}/it_{i+1}_{log_name}'
            p_miner = ProcessMiner(miner_type=miner_type,
                                   metrics=metrics,
                                   log = log,
                                   outpath=path)
            
            p_miner.discover(algorithm_name=optim_name, **optim_params, store=False)
            p_miner.save_csvs()
    collect_and_concatenate_csvs(base_path=outpath, output_csv='csv_completo.csv')

def collect_and_concatenate_csvs(base_path: str, output_csv: str):
    """
    Recorre todas las subcarpetas dentro de `base_path`, busca archivos `results_variables.csv`,
    los concatena y guarda el resultado en un archivo CSV único.

    Args:
        base_path (str): Ruta base donde están las carpetas generadas.
        output_csv (str): Ruta del archivo CSV de salida.
    """
    all_data = []  
    base_path = os.path.join(os.path.abspath('.'), 'out' , base_path)
    output_csv = os.path.join(base_path, output_csv)

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'results_variables.csv':  
                file_path = os.path.join(root, file)
                try:
                    log_name = os.path.basename(os.path.dirname(file_path)).split('_')[-1]
                    df = pd.read_csv(file_path)

                    ## log del que provienen
                    df['log_name'] = log_name
                    
                    ## redondeo a 2 decimales
                    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                    df[numeric_columns] = df[numeric_columns].round(5)
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"Error al leer el archivo {file_path}: {e}")
    if all_data:
        concatenated_df = pd.concat(all_data, ignore_index=True)
        concatenated_df = concatenated_df.drop_duplicates()
        concatenated_df.to_csv(output_csv, index=False)
        print(f"Archivo concatenado guardado en {output_csv}")
    else:
        print("No se encontraron archivos `results_variables.csv` para concatenar.")

        
## TESTING
if __name__ == "__main__":

    """log_closed = ('closed', 'event_logs/Closed/BPI_Challenge_2013_closed_problems.xes')
    log_open = ('open', 'event_logs/Open/BPI_Challenge_2013_open_problems.xes')
    log_financial = ('financial', 'event_logs/Financial/BPI_Challenge_2012.xes')

    log_list = [log_closed, log_open]

    generate_dataset(miner_type='heuristic',
                     metrics='basic',
                     logs = log_list,
                     outpath='prueba_dataset_100iters',
                     iters=100)"""
    
    log_financial = 'event_logs/Financial/BPI_Challenge_2012.xes'

    log = log_financial
    
    p_miner = ProcessMiner(miner_type='heuristic',
                            metrics='quality',
                            log = log)
    
    p_miner.discover(algorithm_name='NSGAII', **parameters.nsgaii_params)
    p_miner.show_pareto_iterations()
    
