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