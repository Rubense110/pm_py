from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicsParameters
from pm4py.algo.discovery.inductive.algorithm import Variants as InductiveVariants
from pm4py.algo.discovery.inductive.variants.imf import IMFParameters

import pm4py

class Heuristic_Parameters():

    # Heuristic miner default params to begin discovery
    base_params = {
        HeuristicsParameters.DEPENDENCY_THRESH.value: 0.5,
        HeuristicsParameters.AND_MEASURE_THRESH.value: 0.65,
        HeuristicsParameters.MIN_ACT_COUNT.value: 1,
        HeuristicsParameters.MIN_DFG_OCCURRENCES.value: 2,
        HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH.value: 0.05,
        HeuristicsParameters.LOOP_LENGTH_TWO_THRESH.value: 0.5,
    }

    # range of posible values for the parameters of the heurisic miner
    param_range = {
        HeuristicsParameters.DEPENDENCY_THRESH : [0, 1],
        HeuristicsParameters.AND_MEASURE_THRESH : [0, 1],
        HeuristicsParameters.MIN_ACT_COUNT: [1, 1000],
        HeuristicsParameters.MIN_DFG_OCCURRENCES: [1, 1000],
        HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH: [0, 1],
        HeuristicsParameters.LOOP_LENGTH_TWO_THRESH: [0, 1]
    }

    # Data type for each parameter
    param_type = {
        HeuristicsParameters.DEPENDENCY_THRESH : float,
        HeuristicsParameters.AND_MEASURE_THRESH : float,
        HeuristicsParameters.MIN_ACT_COUNT: int,
        HeuristicsParameters.MIN_DFG_OCCURRENCES: int,
        HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH: float,
        HeuristicsParameters.LOOP_LENGTH_TWO_THRESH: float
    }

    def adjust_heu_params(self, log):
        log = pm4py.convert_to_dataframe(log)

        max_activity_count = log['concept:name'].value_counts().max()
        #min_activity_count = log['concept:name'].value_counts().min()

        self.param_range[HeuristicsParameters.MIN_ACT_COUNT] = [0, max_activity_count]
        #self.param_range[HeuristicsParameters.MIN_DFG_OCCURRENCES] = [min_activity_count, max_activity_count]

        #print(self.param_range)

class Inductive_Parameters():
    
    base_params = {InductiveVariants : InductiveVariants.IM}
    param_range = {InductiveVariants : [InductiveVariants.IM, InductiveVariants.IMf, InductiveVariants.IMd]}
    param_type  = {InductiveVariants : 'choice'} # TO-DO -> generic Enum class for this kind of choices

