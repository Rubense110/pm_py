import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pm')))

import common
from pm.process_miner import ProcessMiner
from pm.parameters import BaseParametersConfig
from pm.problem import *

def debug_serialization(obj):
    for attr, value in obj.__dict__.items():
        try:
            pickle.dumps(value)
            print(f"'{attr}' es serializable.")
        except Exception as e:
            print(f"'{attr}' NO es serializable: {e}")

# Prueba de serialización
if __name__ == "__main__":
    import pickle

    # Crear una instancia de prueba (sustituye con datos reales)
    parameters_info = BaseParametersConfig()
    problem = PMProblem(miner_name="heuristic", log_path="event_logs/Closed/BPI_Challenge_2013_closed_problems.xes", metrics_name="basic", parameters_info=parameters_info)

    # Serializar la clase
    with open("/tmp/test_problem.pkl", "wb") as f:
        pickle.dump(problem, f)

    # Deserializar la clase
    with open("/tmp/test_problem.pkl", "rb") as f:
        loaded_problem = pickle.load(f)

    print("\nSerialización y deserialización completadas.\n")

    def debug_serialization(obj):
        for attr, value in obj.__dict__.items():
            try:
                pickle.dumps(value)
                print(f"'{attr}' es serializable.")
            except Exception as e:
                print(f"'{attr}' NO es serializable: {e}")

    # Código de prueba
    debug_serialization(problem)