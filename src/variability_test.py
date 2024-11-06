import os 
import time
import re

from process_miner import ProcessMiner

from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.obj import PetriNet

class VariabilityTest():
        
    out_folder = 'src/tests/out/variability'
    log_file = 'src/tests/out/variability/log.csv'

    def __init__(self, miner_type, opt_type, metrics, log, iterations, epochs):
        self.miner_type = miner_type
        self.opt_type = opt_type
        self.metrics = metrics
        self.log = log
        self.iterations = iterations
        self.epochs = epochs

        self.results_set = list()
        self.test_dict = dict()
        self.log_name = os.path.basename(self.log)

        self.local_time = time.strftime("[%Y_%m_%d - %H:%M:%S]", time.localtime())
        self.test()

    def test(self):
        nsgaii_params = {'population_size': 100,
                         'offspring_population_size': 100,
                         'mutation': PolynomialMutation(probability=0.17, distribution_index=20),
                         'crossover': PMXCrossover(probability=1.0),
                         'termination_criterion': StoppingByEvaluations(max_evaluations=self.iterations)}
        start_time= time.time()
        for epoch in range(self.epochs):
            p_miner = ProcessMiner(miner_type=self.miner_type,
                                   metrics = self.metrics,
                                   log = self.log)
            
            p_miner.set_out_folder(self.out_folder)
            p_miner.discover(algorithm_name=self.opt_type, **nsgaii_params,store=False)

            epoch_result = p_miner.opt.get_pareto_front_petri_nets() ##petri nets pareto
            self.results_set.extend(epoch_result)
            self.test_dict[str(epoch)] = epoch_result

        end_time = time.time()

        self.set_unique_petri_nets()

        print(self.results_set, "\n")
        for i in self.results_set:
            print(i, type(i))

        print(len(self.results_set))

        for i in self.results_set:
            print(i, "\n\n")
        self.runtime = str(end_time - start_time)

        #print(self.results_set)
        #print(self.test_dict)
        
        self.save()
        self.save_to_log()


    def save(self):
        outpath = f'{self.out_folder}/{self.local_time}-{self.epochs}-{self.iterations}-{self.log_name}'
        os.makedirs(outpath, exist_ok=True)

        ## save petris for each epoch
        for epoch, petris in self.test_dict.items():
            for index,petri in enumerate(petris):
                gviz = pn_visualizer.apply(petri[0], petri[1], petri[2])
                epoch_path = f'{outpath}/epoch_{epoch}'
                os.makedirs(epoch_path, exist_ok=True)
                pn_visualizer.save(gviz, f'{epoch_path}/petri_pareto_{index}.png')

        ## Save petri set
        for index, petri in enumerate(self.results_set):
            gviz = pn_visualizer.apply(petri[0], petri[1], petri[2])
            set_path = f'{outpath}/set'
            os.makedirs(set_path, exist_ok=True)
            pn_visualizer.save(gviz, f'{set_path}/petri_pareto_{index}.png')

    def save_to_log(self):
        '''
        Stores relevant information about the execution (e.g. runtime, log name, miner type, etc)
        '''
        if os.path.isfile(self.log_file):
            with open(self.log_file, 'a') as log:
                log.write(f'\n{self.local_time};{self.runtime};{self.iterations};{len(self.results_set)};{self.epochs};{self.log_name};{self.miner_type};{self.opt_type};{self.metrics};')
        else:
            with open(self.log_file, 'w') as log:
                log.write('Timestamp,Runtime, Log Name, Miner Type, Opt type, Opt Parameters, Metrics type, Optimal solution')
            self.save_to_log()

    def are_petri_nets_equal(self, net1: PetriNet, net2: PetriNet) -> bool:
        
        # Comparar el número de lugares
        if len(net1.places) != len(net2.places):
            return False

        # Comparar el número de transiciones
        print(len(net1.transitions),len(net2.transitions),"\n\n")
        if len(net1.transitions) != len(net2.transitions):
            return False

        # Comparar el número de arcos
        if len(net1.arcs) != len(net2.arcs):
            return False

        # Comparar los lugares
        net1.places != net2.places

        return True

    def set_unique_petri_nets(self):
        unique_nets = []
        
        for (net,im,fm) in self.results_set:
            if not any(self.are_petri_nets_equal(net, unique_net) for (unique_net,_,_) in unique_nets):
                unique_nets.append((net,im,fm))

        self.results_set = unique_nets

if __name__ == "__main__":

    log = 'event_logs/Closed/BPI_Challenge_2013_closed_problems.xes'

    test = VariabilityTest(miner_type='inductive',
                           opt_type='NSGAII',
                           metrics='basic_conformance',
                           log = log,
                           iterations=10,
                           epochs=2)