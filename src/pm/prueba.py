import pm4py

from pm4py.visualization.petri_net import visualizer as pn_visualizer

# Cargar el log de eventos (puede ser un XES, CSV o dataframe de Pandas)
log = pm4py.read_xes("event_logs/Financial/BPI_Challenge_2012.xes")  # Cambia por tu archivo de log

# Descubrir la Red de Petri usando el algoritmo Alpha Miner
net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(log)

print(net)

gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.save(gviz, 'src/a.png')
