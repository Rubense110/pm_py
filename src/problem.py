import src.metrics as metrics


import random
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.convert import convert_to_petri_net
from jmetal.core.problem import FloatProblem, FloatSolution


class PM_miner_problem(FloatProblem):
    current_iteration = 0

    def __init__(self, miner, log, metrics_obj: metrics.Metrics, parameters_info):
        
        super(PM_miner_problem, self).__init__()

        self.miner = miner                       # pm4py miner e.g. heuristic_miner
        self.log = log                           # XES log (already loaded by pm4py)
        self.metrics_obj = metrics_obj           # How to calculate fitness
        self.parameters_info = parameters_info   # Miner pararameters (selected automatically)

        self.number_of_objectives = metrics_obj.get_n_of_metrics()
        self.number_of_variables = self.__get_n_genes()
        self.number_of_constraints = 0

        self.lower_bound, self.upper_bound = self.__get_bounds()

        #self.obj_directions = [self.MINIMIZE]
        self.obj_labels = metrics_obj.get_labels()

    # Determines the value range for the parameters of the specified miner
    def __get_bounds(self):
        lower_bound = [i[0] for i in self.parameters_info.param_range.values()]
        upper_bound = [i[1] for i in self.parameters_info.param_range.values()]
        return lower_bound, upper_bound
    
    def __get_n_genes(self):
        return len(self.parameters_info.param_range)
        
    def evaluate(self, solution: FloatSolution) -> FloatSolution :
        params = {key: solution.variables[idx] for idx, key in enumerate(self.parameters_info.param_range.keys())}
        petri, _, _ = self._create_petri_net_sol(params)
        solution.objectives = self.metrics_obj.get_metrics_array(petri)
        solution.number_of_objectives = self.number_of_objectives
        
        return solution
    
    def _create_petri_net_sol(self, params):
        if self.miner == heuristics_miner:
            petri, initial_marking, final_marking = self.miner.apply(self.log, parameters= params)
        elif self.miner == inductive_miner:
            inductive_variant = inductive_miner.Variants.IMf if params["noise_threshold"] > 0 else inductive_miner.Variants.IM
            process_tree = self.miner.apply(self.log, variant = inductive_variant,  parameters= params )
            petri, initial_marking, final_marking = convert_to_petri_net(process_tree)    
        return petri, initial_marking, final_marking
    
    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(number_of_constraints=self.number_of_constraints,
                                     number_of_objectives=self.number_of_objectives,
                                     lower_bound = self.lower_bound,
                                     upper_bound = self.upper_bound)   
        # Random Solution
        random_sol = list()
        for index, param_and_type in enumerate(self.parameters_info.param_type.items()):
            data_type = param_and_type[1]
            if data_type == int: 
                random_sol.append(random.randint(self.lower_bound[index], self.upper_bound[index]))
            elif data_type == bool:
                random_sol.append(random.choice([True, False]))
            else:
                random_sol.append(random.uniform(self.lower_bound[index], self.upper_bound[index]))


        new_solution.variables = random_sol
        return new_solution

    def name(self) -> str:
        return 'Custom Process Mining Problem'
    
    def number_of_constraints(self) -> int:
        return super().number_of_constraints()
    
    def number_of_variables(self) -> int:
        return super().number_of_variables()
    
    def number_of_objectives(self) -> int:
        return super().number_of_objectives()