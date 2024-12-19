from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import parameters
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import os

import pandas as pd
# Ajustar las opciones de visualización
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000) 
pd.set_option('display.colheader_justify', 'left') 

log_mapping = {
    'log_name_open' : 'event_logs/Closed/BPI_Challenge_2013_closed_problems.xes',
    'log_name_closed' : 'event_logs/Open/BPI_Challenge_2013_open_problems.xes'
}

def predict_one_class(csv_path, outpath):

    outpath = f'src/tests/anomalas/{outpath}'
    outpath_petris = outpath+'/petris'

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(outpath_petris, exist_ok=True)

    param_names = parameters.HeuristicParametersConfig()

    data = pd.read_csv(csv_path)
    data = pd.get_dummies(data, columns=['log_name'], prefix='log_name')
    X = data

    #model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X)
    predictions = model.predict(X)
    print(predictions, "\n")

    anomalous_rows = data[predictions == -1]
    print(anomalous_rows)

    for index, row in anomalous_rows.iterrows():
        params_values = row.tolist() 
        log_columns = [col for col in row.index if col.startswith('log_name_') and row[col] == 1][0]
        params = {key: params_values[idx] for idx, key in enumerate(parameters.HeuristicParametersConfig.param_range.keys())}
        log = xes_importer.apply(log_mapping[log_columns])
        petri, im, fm = heuristics_miner.apply(log, parameters= params)

        gviz = pn_visualizer.apply(petri, im, fm)
        pn_visualizer.save(gviz, f'{outpath}/petri_anomala_{index}.png')

    for index, row in data.iterrows():
        params_values = row.tolist() 
        log_columns = [col for col in row.index if col.startswith('log_name_') and row[col] == 1][0]
        params = {key: params_values[idx] for idx, key in enumerate(param_names.param_range.keys())}
        log = xes_importer.apply(log_mapping[log_columns])
        petri, im, fm = heuristics_miner.apply(log, parameters= params)
        gviz = pn_visualizer.apply(petri, im, fm)
        pn_visualizer.save(gviz, f'{outpath}/petris/petri_{index}.png')

if __name__ == '__main__':

    predict_one_class('src/csv_completo.csv', 'pruebas')
