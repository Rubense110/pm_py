## Known Issues 

### Fixed

- `jmetal.util.archive.py` -> Changed `NonDominatedSolutionsArchive.add()` to be able to compare numpy arrays (.all())

    ```python
    # line 105
    if solution.objectives.all() == current_solution.objectives.all():
    ```

- `jmetal.operator.crossover.py` -> Changed `SBXCrossover.execute()` to prevent complex numbers from ruining the execution.

    ```python
    # line 193
    if isinstance(c1, complex):
        c1 = c1.real
    if isinstance(c2, complex):
        c2 = c2.real
    ```

### Clustering

    Se usarán los OBJETIVOS de las soluciones para agrupar. Trabajamos entonces con el csv y ya.

    n_places,n_arcs,n_transitions,cycl_complx,ratio,joins,splits
    
    3.0,4.0,2.0,1.0,1.5,1.0,0.0
    17.0,72.0,36.0,21.0,0.4722222222222222,7.0,8.0
    16.0,66.0,33.0,19.0,0.48484848484848486,7.0,7.0

    Distancia -> Manhattan distance
    algoritmo -> Single-Linkage Agglomerative Clustering

    Averaged Hausdorff Distance (AHD) -> cómo de cerca están dos soluciones en funcion de Manhattan.
    Minimum Pairwise Distance (MPD) -> medir la no similaridad entre los modelos devueltos tras el clustering.

--- 

activate venv -> .\.venv\Scripts\activate --- source .venv/bin/activate
PATH -> export PYTHONPATH="${PYTHONPATH}:/home/ruben/Documents/TFG/" ## echo $PYTHONPATH para verlo

---
### Imports

`sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`

- `os.path.dirname(__file__)`: Devuelve la ruta del directorio donde se encuentra process_miner.py.
- `os.path.join(os.path.dirname(__file__), '..')`: Navega un nivel hacia atrás desde pm/ para acceder a src/.
- `sys.path.append(...)`: Añade el directorio src/ al sys.path, permitiendo importar módulos que están a ese nivel.

### Parallel

`/.venv/lib/python3.11/site-packages/jmetal/util/evaluator.py` -> tocado.

Para verificar serialización:
```python
# Función a alterar.
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        #mio
        def debug_serialization(obj, obj_name):
            """
            Debugs the serialization of an object and its attributes.

            Parameters:
            - obj: The object to test for serialization.
            - obj_name: A name to identify the object in debug messages.
            """
            try:
                pickle.dumps(obj)
                print(f"'{obj_name}' es serializable.")
            except Exception as e:
                print(f"'{obj_name}' NO es serializable: {e}")
                
                # Inspect attributes of the object if it is not serializable
                if hasattr(obj, '__dict__'):
                    print(f"--- Analizando atributos de '{obj_name}' ---")
                    for attr_name, attr_value in obj.__dict__.items():
                        try:
                            pickle.dumps(attr_value)
                            print(f"  El atributo '{attr_name}' es serializable.")
                        except Exception as attr_error:
                            print(f"  El atributo '{attr_name}' NO es serializable: {attr_error}")
                else:
                    print(f"'{obj_name}' no tiene un atributo __dict__ para inspeccionar.")


        debug_serialization(problem, "problem")
        debug_serialization(solution_list, "solution_list")
        #finmio
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)
```