import src.metrics as metrics

from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicsParameters
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
import pm4py
from abc import abstractmethod


class Base_Parameters():
    base_params = {}
    param_range = {}
    param_type = {}


class Heuristic_Parameters(Base_Parameters):

    def __init__(self, log):
        super().__init__()
            # Heuristic miner default params to begin discovery
        self.base_params = {
            HeuristicsParameters.DEPENDENCY_THRESH.value: 0.5,
            HeuristicsParameters.AND_MEASURE_THRESH.value: 0.65,
            HeuristicsParameters.MIN_ACT_COUNT.value: 1,
            HeuristicsParameters.MIN_DFG_OCCURRENCES.value: 2,
            HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH.value: 0.05,
            HeuristicsParameters.LOOP_LENGTH_TWO_THRESH.value: 0.5,
        }

        # range of posible values for the parameters of the heurisic miner
        self.param_range = {
            HeuristicsParameters.DEPENDENCY_THRESH : [0, 1],
            HeuristicsParameters.AND_MEASURE_THRESH : [0, 1],
            HeuristicsParameters.MIN_ACT_COUNT: [1, 1000],
            HeuristicsParameters.MIN_DFG_OCCURRENCES: [1, 1000],
            HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH: [0, 1],
            HeuristicsParameters.LOOP_LENGTH_TWO_THRESH: [0, 1]
        }

        # Data type for each parameter
        self.param_type = {
            HeuristicsParameters.DEPENDENCY_THRESH : float,
            HeuristicsParameters.AND_MEASURE_THRESH : float,
            HeuristicsParameters.MIN_ACT_COUNT: int,
            HeuristicsParameters.MIN_DFG_OCCURRENCES: int,
            HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH: float,
            HeuristicsParameters.LOOP_LENGTH_TWO_THRESH: float
        }
        self.adjust_heu_params(log)

    def adjust_heu_params(self, log):
        log = pm4py.convert_to_dataframe(log)

        max_activity_count = log['concept:name'].value_counts().max()
        #min_activity_count = log['concept:name'].value_counts().min()

        self.param_range[HeuristicsParameters.MIN_ACT_COUNT] = [0, max_activity_count]
        #self.param_range[HeuristicsParameters.MIN_DFG_OCCURRENCES] = [min_activity_count, max_activity_count]

        #print(self.param_range)

class Inductive_Parameters(Base_Parameters):
    def __init__(self) -> None:
        super().__init__()
    
        self.base_params = {'noise_threshold' : 0,
                            'multi_processing': False,
                            'disable_fallthroughs': False}
        
        self.param_range = {'noise_threshold' : [0, 1],
                            'multi_processing': [0, 1],
                            'disable_fallthroughs': [0, 1]}
        
        self.param_type  = {'noise_threshold' : float,
                            'multi_processing': float,
                            'disable_fallthroughs': float}
            
miner_mapping = {
    'inductive': inductive_miner, 
    'heuristic': heuristics_miner,
}

metrics_mapping = {
    'basic': metrics.Basic_Metrics(),
    'basic_useful_simple': metrics.Basic_Metrics_Usefull_simple()
}

