    
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_petri_net


# Cargar un log de eventos desde un archivo XES
log = xes_importer.apply('test/Financial/BPI_Challenge_2012.xes')

# obtain optimal petri net
pt = inductive_miner.apply(log)
net, initial_marking, final_marking= convert_to_petri_net(pt)
# visualize petri net
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)