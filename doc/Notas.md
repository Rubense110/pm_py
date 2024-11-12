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