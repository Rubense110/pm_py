from abc import abstractmethod
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.conformance import fitness_alignments
from pm4py.conformance import precision_alignments
from pm4py.conformance import fitness_token_based_replay
from pm4py.conformance import precision_token_based_replay
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

    def get_metrics_array(self, petri: PetriNet, im, fm, log):

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
    
class Basic_Conformance(Basic_Metrics):

    def __init__(self):
        super().__init__()

        fitness = 0
        precission = 0

        self.metrics_array = np.append(self.metrics_array,[fitness, precission])
        self.n_of_metrics = len(self.metrics_array)
        self.labels.extend(['fitness', 'precission'])

    def get_metrics_array(self, petri, im, fm, events_log):
        basic_metrics =  super().get_metrics_array(petri, im, fm, events_log)

        #fitness_alignment_dict = fitness_alignments(log=events_log, petri_net=petri, initial_marking=im, final_marking=fm)
        #fitness_alignment = fitness_alignment_dict['average_trace_fitness']

        fitness_token_dict = fitness_token_based_replay(log=events_log, petri_net=petri, initial_marking=im, final_marking=fm)
        fitness_token = fitness_token_dict['average_trace_fitness']

        #precission_alignment = precision_alignments(log=events_log, petri_net=petri, initial_marking=im, final_marking=fm)
        precission_token = precision_token_based_replay(log=events_log, petri_net=petri, initial_marking=im, final_marking=fm)


        basic_metrics = np.append(basic_metrics, [-fitness_token, -precission_token]) ## estamos minimizando 
        self.metrics_array = basic_metrics
        return self.metrics_array

class Quality_Metrics(Metrics):

    def __init__(self):
        super().__init__()

        fitness = 0
        precision = 0
        simplicity = 0
        generalization = 0

        self.metrics_array = np.array([fitness, precision, simplicity, generalization])
        self.n_of_metrics = len(self.metrics_array)
        self.labels = ['fitness', 'precision', 'simplicity', 'generalization']

    def get_metrics_array(self, petri: PetriNet, im, fm, events_log):
        
        fitness_token_dict = fitness_token_based_replay(log=events_log, petri_net=petri, initial_marking=im, final_marking=fm)
        fitness = fitness_token_dict['average_trace_fitness']
        precision = precision_token_based_replay(log=events_log, petri_net=petri, initial_marking=im, final_marking=fm)
        simplicity = self.__calculate_simplicity(petri)
        generalization = self.__calculate_generalization(petri, events_log)

        self.metrics_array = np.array([fitness, precision, simplicity, generalization])
        return self.metrics_array

    def get_n_of_metrics(self):
        return self.n_of_metrics

    def get_labels(self):
        return self.labels
    
    def __calculate_simplicity(self, petri: PetriNet):
        """
        Calculate the simplicity metric as per the given formula.
        ∂s = (|T| − |TDT| + |TIT|) / |T|

        T: Total transitions
        TDT: Alternative duplicate tasks
        TIT: Redundant invisible tasks
        """

        transitions = petri.transitions
        T = len(transitions)
        TDT = self.__count_alternative_duplicate_tasks(transitions)
        TIT = self.__count_redundant_invisible_tasks(transitions)

        if T == 0:  
            return 0
        return (T - TDT + TIT) / T
    
    def __count_alternative_duplicate_tasks(self, transitions):
        """
        Count the number of alternative duplicate tasks (TDT).
        These are tasks that are never repeated simultaneously within a single sequence.
        """
        task_names = {}
        TDT = 0

        for transition in transitions:
            if not transition.label:  # Skip invisible transitions
                continue
            if transition.label not in task_names:
                task_names[transition.label] = 1
            else:
                task_names[transition.label] += 1

        # Count tasks that appear more than once
        for count in task_names.values():
            if count > 1:
                TDT += 1
        return TDT
    
    def __count_redundant_invisible_tasks(self, transitions):
        """
        Count the number of redundant invisible tasks (TIT).
        These are invisible tasks that can be safely eliminated from the model without altering its behavior.
        """
        TIT = 0

        for transition in transitions:
            if transition.label is None or transition.label == "":
                # Check if the invisible task is redundant (no arcs or same input/output places)
                if transition.in_arcs == transition.out_arcs:
                    TIT += 1

        return TIT
    
    def __calculate_generalization(self, petri: PetriNet, events_log):
        """
        Calculate the generalization metric as per the given formula:
        Qg = 1 − (Σnodes(√#execution)−1) / 'nodes_in_model'
        """
        # Create a mapping of transitions and their execution counts from the event log
        execution_counts = self.__get_transition_execution_counts(events_log, petri.transitions)

        total_nodes = len(petri.transitions)
        if total_nodes == 0:  # Avoid division by zero
            return 0

        generalization_sum = 0
        for transition in petri.transitions:
            # Execution count for the transition
            execution_count = execution_counts.get(transition.label, 0)
            generalization_sum += (1 / (1 + np.sqrt(execution_count)))

        # Generalization score
        return 1 - (generalization_sum / total_nodes)

    def __get_transition_execution_counts(self, events_log, transitions):
        """
        Create a mapping of transition labels to their execution counts from the event log.
        """
        execution_counts = {t.label: 0 for t in transitions if t.label}

        for trace in events_log:
            for event in trace:
                if event["concept:name"] in execution_counts:
                    execution_counts[event["concept:name"]] += 1

        return execution_counts
    
class Basic_Metrics_Usefull_Simple(Basic_Metrics):

    ## RULES
    NSFE_TRESHOLD_MORE_THAN = 1.5
    NSFE_TRESHOLD_LESS_THAN = 0.5
    GM_TRESHOLD_MORE_THAN = 1.5
    
    def __init__(self):
        super().__init__()
        
        ## TO-DO -> add this info to the logs
        #basic_useful_metrics_labels = ["is_ap", "NSFE", "GM"]
        #self.labels.extend(basic_useful_metrics_labels)

    def get_metrics_array(self, petri: PetriNet, im, fm, log):
        petri_graph = self.__convert_petri_to_graph(petri)
        #self.__plot_petri_graph(petri_graph)

        NSFE = self.__calculate_nsfe(petri)
        IS_AP = nx.is_aperiodic(petri_graph)
        GM = self.__calculate_gm(petri)

        #print(NSFE, IS_AP, GM)

        if  IS_AP or  self.__check_nsfe_interval(NSFE) or  self.__check_gm_interval(GM):
            return super().get_metrics_array(petri, im, fm, log)
        else:
            penalty = 1e6
            return np.array([penalty] * self.n_of_metrics)
            

    def __check_nsfe_interval(self, NSFE):
        if NSFE <= self.NSFE_TRESHOLD_LESS_THAN or NSFE > self.NSFE_TRESHOLD_MORE_THAN:
            return True
        else:
            return False
    def __check_gm_interval(self, GM):
        if GM > self.GM_TRESHOLD_MORE_THAN:
            return True
        else:
            return False
    
    def __convert_petri_to_graph(self, petri: PetriNet):
        graph = nx.DiGraph()

        for place in petri.places:
            graph.add_node(place, type= "place")

        for transition in petri.transitions:
            graph.add_node(transition, type= "transition")

        for arc in petri.arcs:
            graph.add_edge(arc.source, arc.target)

        return graph
    
    def __calculate_gm(self, petri: PetriNet):
        '''Overall difference between incoming and outgoing'''
        gm = 0
        for place in petri.places:
            in_count = len(place.in_arcs)
            out_count = len(place.out_arcs)
            gm += abs(in_count - out_count)
        gm = gm//len(petri.places)
        return gm

    def __calculate_nsfe(self, petri: PetriNet):
        '''Total number of outgoing flows from events'''
        nsfe = 0
        for places in petri.places:
            nsfe += len(places.out_arcs)
        nsfe = nsfe/len(petri.places) # Normalize
        return nsfe
        
    ## -- Testing purposes --
    def __plot_petri_graph(self, graph: nx.DiGraph):
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=10, font_weight='bold')
        plt.savefig("petrigraph.png")

class Extended_Quality_Metrics(Quality_Metrics):

    def __init__(self):
        super().__init__()

        # Adding new metrics for distances
        fpd = 0  # Fitness-Precision Distance
        sgd = 0  # Simplicity-Generalization Distance

        self.metrics_array = np.append(self.metrics_array, [fpd, sgd])
        self.n_of_metrics = len(self.metrics_array)
        self.labels.extend(['fp_distance', 'sg_distance'])

    def get_metrics_array(self, petri: PetriNet, im, fm, events_log):
        # Calculate base quality metrics
        base_metrics = super().get_metrics_array(petri, im, fm, events_log)

        # Calculate distances
        fitness = base_metrics[0]  # Fitness
        precision = base_metrics[1]  # Precision
        simplicity = base_metrics[2]  # Simplicity
        generalization = base_metrics[3]  # Generalization

        fpd = abs(fitness - precision)
        sgd = abs(simplicity - generalization)

        # Add new metrics to array
        extended_metrics = np.append(base_metrics, [fpd, sgd])
        self.metrics_array = extended_metrics
        return self.metrics_array

    def get_labels(self):
        return self.labels

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

    metrics_obj =  Basic_Metrics_Usefull_Simple()
    metrics_labels = metrics_obj.get_labels()
    #metrics = 
    print(metrics_obj.get_metrics_array(net))

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
