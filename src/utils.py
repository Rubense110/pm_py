import numpy as np
from jmetal.core.problem import FloatSolution
import matplotlib.pyplot as plt

def calculate_pareto_front(solution_objectives):
    solution_objectives = np.array(solution_objectives)
    front = []

    for i, sol1 in enumerate(solution_objectives):
        dominated = False
        for j, sol2 in enumerate(solution_objectives):
            if i!=j:
                if np.all(sol2 <= sol1) and np.any(sol2 < sol1):
                    dominated = True
                    break
        if not dominated and not any(np.array_equal(sol1, f) for f in front):
            front.append(sol1)

    return np.array(front)

def plot_pareto_front(objectives, ):
    objectives = np.array(objectives)
    pareto_front = calculate_pareto_front(objectives)

    for solution in objectives:
        print(solution)
        plt.plot(solution.shape[0], solution, color= 'blue')

    for optimal_solution in pareto_front:
        plt.plot(len(optimal_solution), optimal_solution, color= 'red')

    plt.xlabel('Objetivo 1')
    plt.ylabel('Objetivo 2')
    plt.title('Soluciones y Frente de Pareto')
    plt.legend()

    plt.savefig('pareto.png')

if __name__ == "__main__":
    import pandas as pd
    opt_sols_data = pd.read_csv('out/[2024_09_13 - 18:46:01]-BPI_Challenge_2013_closed_problems.xes-NSGA-II/results_objectives.csv')
    pareto_front = calculate_pareto_front(opt_sols_data.values.tolist())
    print(pareto_front)
    #for i in pareto_front:
    #    print(i)

    plot_pareto_front(opt_sols_data.values.tolist())

