from abc import abstractmethod
from pm4py.objects.petri_net.obj import PetriNet
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Metrics():

    def __init__(self: PetriNet):
        self.n_of_metrics = 0
        self.metrics_array = np.array([])
        self.labels = []

    @abstractmethod
    def get_n_of_metrics(self):
        pass

    @abstractmethod
    def get_metrics_array(self):
        pass

    @abstractmethod
    def get_labels(self):
        pass

    def _get_joins_splits(self,arcs):

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


class Basic_Metrics(Metrics):

    def __init__(self):
        super(Metrics, self).__init__()

        n_places = 0
        n_arcs = 0
        n_transitions = 0
        cycl_complx = 0
        ratio = 0
        joins = 0
        splits = 0

        
        self.metrics_array = np.array([n_places, n_arcs, n_transitions, cycl_complx, ratio, joins, splits])
        self.n_of_metrics = len(self.metrics_array)
        self.labels = ['n_places', 'n_arcs', 'n_transitions', 'cycl_complx', 'ratio', 'joins', 'splits']

    def get_metrics_array(self, petri: PetriNet):

                # basic metrics
        n_places = len(petri.places)
        n_transitions = len(petri.transitions)
        n_arcs = len(petri.arcs)

        # Cyclomatic complexity
        cycl_complx = n_arcs - (n_places + n_transitions) + 2

        #ratio states/transition
        ratio = n_places/n_transitions

        #joins & splits
        joins, splits = self._get_joins_splits(petri.arcs)

        # -joins & -splits if Maximize, joins & splits if Minimize
        self.metrics_array = np.array([n_places, n_arcs, n_transitions, cycl_complx, ratio, joins, splits])
        return self.metrics_array
    
    def get_n_of_metrics(self):
        return self.n_of_metrics
    
    def get_labels(self):
        return self.labels

class Basic_Metrics_Usefull_simple(Basic_Metrics):
    
    def __init__(self):
        super(Basic_Metrics, self).__init__()

        is_ap = False
        NSFE = 0
        GM = 0
        basic_useful_metrics = [is_ap, NSFE, GM]
        basic_useful_metrics_labels = ["is_ap", "NSFE", "GM"]

        self.metrics_array = np.append(self.get_metrics_array, np.array(basic_useful_metrics))
        self.n_of_metrics = len(self.metrics_array)
        self.labels.extend(basic_useful_metrics_labels)

    def get_metrics_array(self, petri: PetriNet):
        basic_metrics = super().get_metrics_array(petri)
        petri_graph = self.__convert_to_graph(petri)
        #self.__plot_petri_graph(petri_graph)

        is_ap = nx.is_aperiodic(petri_graph)


    def __plot_petri_graph(self, graph: nx.DiGraph):
        ## -- Testing purposes --
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=10, font_weight='bold')
        plt.savefig("petrigraph.png")

    def __convert_to_graph(self, petri: PetriNet):
        graph = nx.DiGraph()

        for place in petri.places:
            graph.add_node(place, type= "place")

        for transition in petri.transitions:
            graph.add_node(transition, type= "transition")

        for arc in petri.arcs:
            graph.add_edge(arc.source, arc.target)

        return graph
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

    metrics_obj =  Basic_Metrics_Usefull_simple()
    metrics_labels = metrics_obj.get_labels()
    #metrics = 
    metrics_obj.get_metrics_array(net)

    def show(metrics):
        print(type(metrics), metrics)
        print("\n### METRICS ###")
        print(f"{metrics_labels[0]}:    {metrics[0]}")
        print(f"{metrics_labels[1]}:    {metrics[1]}")
        print(f"{metrics_labels[2]}:    {metrics[2]}")
        print(f"{metrics_labels[3]}:    {metrics[3]}")
        print(f"{metrics_labels[4]}:    {metrics[4]}")
        print(f"{metrics_labels[5]}:    {metrics[5]}")
        print(f"{metrics_labels[6]}:    {metrics[6]}\n")
        print(f"Nº of metrics: {metrics_obj.get_n_of_metrics()}")
