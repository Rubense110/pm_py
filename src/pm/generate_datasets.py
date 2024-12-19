import optimize
import parameters
from process_miner import ProcessMiner

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal_fixed import NSGAIII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import time
import os
import pandas as pd

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

if __name__ == "__main__": 

    log_closed = ('closed', 'event_logs/Closed/BPI_Challenge_2013_closed_problems.xes')
    log_open = ('open', 'event_logs/Open/BPI_Challenge_2013_open_problems.xes')
    log_financial = ('financial', 'event_logs/Financial/BPI_Challenge_2012.xes')

    log_list = [log_closed, log_open]

    generate_dataset(miner_type='heuristic',
                     metrics='basic',
                     logs = log_list,
                     outpath='prueba_dataset_100iters',
                     iters=100)