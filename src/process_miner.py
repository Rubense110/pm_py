# activate venv -> .\.venv\Scripts\activate

import pm4py
from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicsParameters

import params

class Process_miner:

    miner_mapping = {
        #'inductive': pm4py.algo.discovery.inductive,
        'heuristic': pm4py.algo.discovery.heuristics,
        #'alpha': pm4py.algo.discovery.alpha
    }

    def __init__(self, miner_type, log, conformance):
        self.miner = self.__get_miner_alg(miner_type)
        self.log = log
        self.conformance_type = conformance
        self.params = self.__init_params(miner_type)

    def __get_miner_alg(self, miner):
        if miner not in self.miner_mapping:
            raise ValueError(f"Minero '{miner}' no está soportado. Los mineros disponibles son: {list(self.miner_mapping.keys())}")
        return self.miner_mapping[miner]
    
    def __init_params(self, miner_type):
        if miner_type== 'heuristic':
            params = params.base_heu_params

    def __update_params(self, miner_type, **kwargs):
        if miner_type== 'heuristic':
            params = {param.value: kwargs.get(param.value, None) for param in HeuristicsParameters}
        else:
            raise ValueError(f"Unsupported miner type: {miner_type}")
        return params

    def discover(self):
        event_log =  pm4py.read_xes(self.log)
        net, im, fm = self.miner.apply(event_log, **params)


if __name__ == "__main__":
    pass
