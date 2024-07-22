from pm4py.objects.petri_net.obj import PetriNet
import numpy as np

N_BASE_METRICS = 7

def get_base_metrics(petri: PetriNet):

    # basic metrics
    n_places = len(petri.places)
    n_transitions = len(petri.transitions)
    n_arcs = len(petri.arcs)

    # Cyclomatic complexity
    cycl_complx = n_arcs - (n_places + n_transitions) + 2

    #ratio states/transition
    ratio = n_places/n_transitions

    #joins & splits
    joins, splits = get_joins_splits(petri.arcs)

    # -joins & -splits if Maximize, joins & splits if Minimize
    return np.array([n_places, n_arcs, n_transitions, cycl_complx, ratio, joins, splits]) 

def get_joins_splits(arcs):

    splits = dict()
    joins = dict()
    n_splits = 0
    n_joins = 0

    for arc in arcs:

        # obtain split dict
        if arc.source not in splits:
            splits[arc.source] = [arc.target]
        else:
            splits[arc.source].append(arc.target)

        # obtain join dict
        if arc.target not in joins:
            joins[arc.target] = [arc.source]
        else:
            joins[arc.target].append(arc.source)

    # count joins and splits
    for i in splits.values():
        if len(i) > 1: n_splits+=1

    for i in joins.values():
        if len(i) > 1: n_joins+=1

    return (n_splits, n_joins)

## TESTING
if __name__ == "__main__":

    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer



    log = xes_importer.apply('test/Closed/BPI_Challenge_2013_closed_problems.xes')
    net, initial_marking, final_marking = heuristics_miner.apply(log)

    # visualize petri net
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.view(gviz)

    metrics =  get_base_metrics(net)

    print("\n### METRICS ###")
    print(f"nº of places:               {metrics[0]}")
    print(f"nº of arcs:                 {metrics[1]}")
    print(f"nº of transitions:          {metrics[2]}")
    print(f"cyclomatic complexity       {metrics[3]}")
    print(f"ratio places/transitions    {metrics[4]}")
    print(f"nº of joins:                {metrics[5]}")
    print(f"nº of splits:               {metrics[6]}\n")
