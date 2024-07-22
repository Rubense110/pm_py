# activate venv -> .\.venv\Scripts\activate

import pm4py
from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicsParameters

import parameters
import optimize

class Process_miner:

    miner_mapping = {
        #'inductive': pm4py.algo.discovery.inductive,
        'heuristic': pm4py.algo.discovery.heuristics,
        #'alpha': pm4py.algo.discovery.alpha
    }

    def __init__(self, miner_type, log):
        self.miner = self.__get_miner_alg(miner_type)
        self.log = pm4py.read_xes(self.log)
        self.params = self.__init_params(miner_type)

    def __get_miner_alg(self, miner):
        if miner not in self.miner_mapping:
            raise ValueError(f"Minero '{miner}' no está soportado. Los mineros disponibles son: {list(self.miner_mapping.keys())}")
        return self.miner_mapping[miner]
    
    def __init_params(self, miner_type):
        if miner_type== 'heuristic':
            params = parameters.base_heu_params
        return params

    def __update_params(self, miner_type, **kwargs):
        if miner_type== 'heuristic':
            params = {param.value: kwargs.get(param.value, None) for param in HeuristicsParameters}
        else:
            raise ValueError(f"Unsupported miner type: {miner_type}")
        return params

    def discover(self):
        optimize.genetics(self.miner, self.log, self.params)


if __name__ == "__main__":
        
    from pm4py.objects.log.importer.xes import importer as xes_importer


    log = xes_importer.apply('test/Closed/BPI_Challenge_2013_closed_problems.xes')
    pm = Process_miner('heuristic', log)
    print(pm.params)