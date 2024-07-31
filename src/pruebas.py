from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import parameters
import numpy as np
import pandas as pd



log = xes_importer.apply('test/Financial/BPI_Challenge_2012.xes')
base_params = parameters.Heuristic_Parameters.base_params
params = [0.5836558245697305, 0.21489094849903473, np.float64(18376.16368557422), 6.443691641453263, 0.9946396111587875, 0.4529568498101284]
params = [0.9540982738419803, 0.8323425979823712, 54840, 6, 0.6518876642720254, 0.23811459344233893]
params = [0.9999525676553562, 0.9149726963715235, np.float64(1189.8006990838908), 3.0328536280343013, 0.014722361202136103, 0.9294802030455518]


def params_petri(params):

    disc_params = dict()
    for (i,j) in zip(base_params.keys(), params):
        disc_params[i] = j
    print(disc_params)

    petri, im, fm = heuristics_miner.apply(log, disc_params)
    gviz = pn_visualizer.apply(petri, im, fm)
    pn_visualizer.view(gviz)

df = pd.read_csv('doc/log.csv', delimiter=";")
print(df.head())