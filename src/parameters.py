from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicsParameters

# Heuristic miner default params to begin discovery
base_heu_params = {
    HeuristicsParameters.DEPENDENCY_THRESH.value: 0.5,
    HeuristicsParameters.AND_MEASURE_THRESH.value: 0.65,
    HeuristicsParameters.MIN_ACT_COUNT.value: 1,
    HeuristicsParameters.MIN_DFG_OCCURRENCES.value: 2,
    HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH.value: 0.05,
    HeuristicsParameters.LOOP_LENGTH_TWO_THRESH.value: 0.5,
}

# range of posible values for the parameters of the heurisic miner
heu_param_range = {
    HeuristicsParameters.DEPENDENCY_THRESH : [0, 1],
    HeuristicsParameters.AND_MEASURE_THRESH : [0, 1],
    HeuristicsParameters.MIN_ACT_COUNT: [1, 1000],
    HeuristicsParameters.MIN_DFG_OCCURRENCES: [1, 1000],
    HeuristicsParameters.DFG_PRE_CLEANING_NOISE_THRESH: [0, 1],
    HeuristicsParameters.LOOP_LENGTH_TWO_THRESH: [0, 1]
}