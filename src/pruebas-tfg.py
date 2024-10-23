# Importar PM4Py
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.convert import convert_to_petri_net

# 1. Cargar el log XES
log = xes_importer.apply('event_logs/Closed/BPI_Challenge_2013_closed_problems.xes')

process_tree = inductive_miner.apply(log)
petri, initial_marking, final_marking = convert_to_petri_net(process_tree) 
gviz_petri = pn_visualizer.apply(petri, initial_marking, final_marking)
pn_visualizer.view(gviz_petri)

# Si deseas visualizar el modelo de Petri derivado del minero inductivo
petri_net, initial_marking, final_marking = heuristics_miner.apply(log)
gviz_petri = pn_visualizer.apply(petri_net, initial_marking, final_marking)
pn_visualizer.view(gviz_petri)
