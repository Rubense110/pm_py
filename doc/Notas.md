## Codigo

- Mostrar parámetros minero **Heuristico**

```py
    import pm4py
    from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicsParameters

    def get_miner_parameters(miner_type):
        if miner_type == 'heuristic':
            return {param.value: None for param in HeuristicsParameters}
        else:
            raise ValueError(f"Unsupported miner type: {miner_type}")

    selected_miner = 'heuristic'

    params = get_miner_parameters(selected_miner)

    print(f"Parameters for {selected_miner} miner:")
    for key, value in params.items():
        print(f"{key}: {value}")
```

- ver los posibles parametros de una función

```py
    #parametros
    print(pm4py.algo.discovery.heuristics.algorithm.apply_heu.__code__.co_varnames)
```

## Papers

- [Paper HeuristicMiner](https://www.researchgate.net/publication/306014995_Process_mining_with_the_heuristics_miner-algorithm)

--- 


**Discovery** (petri):

todos requieren:
	- log
	- act_key
	- timest_key
	- caseId_key

- alpha.
- inductive ->
	- - **noise_threshold** (`float`) – noise threshold (default: 0.0)
	- - **multi_processing** (`bool`) – boolean that enables/disables multiprocessing in inductive miner
	- - **disable_fallthroughs** (`bool`) – disable the Inductive Miner fall-throughs
	-
- heuristic ->
	- - **dependency_threshold** (`float`) – dependency threshold (default: 0.5)
	- - **and_threshold** (`float`) – AND threshold (default: 0.65)
	- - **loop_two_threshold** (`float`) – loop two threshold (default: 0.5)
	- **Completos**
		- Parameters.ACTIVITY_KEY
		- Parameters.TIMESTAMP_KEY
		- Parameters.CASE_ID_KEY
		- Parameters.DEPENDENCY_THRESH
		- Parameters.AND_MEASURE_THRESH
		- Parameters.MIN_ACT_COUNT
		- Parameters.MIN_DFG_OCCURRENCES
		- Parameters.DFG_PRE_CLEANING_NOISE_THRESH
		- Parameters.LOOP_LENGTH_TWO_THRESH

- ILP ->  
	- - **alpha** (`float`) – noise threshold for the sequence encoding graph (1.0=no filtering, 0.0=greatest filtering)
	- 




**fitness**:
- token based -> [`pm4py.conformance.fitness_token_based_replay()`](https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.11/pm4py.html#pm4py.conformance.fitness_token_based_replay "pm4py.conformance.fitness_token_based_replay")
- alignments -> [`pm4py.conformance.fitness_alignments()`](https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.11/pm4py.html#pm4py.conformance.fitness_alignments "pm4py.conformance.fitness_alignments")
- footprints -> [`pm4py.conformance.fitness_footprints()`](https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.11/pm4py.html#pm4py.conformance.fitness_footprints "pm4py.conformance.fitness_footprints")

**Precission**:
- token based -> [`pm4py.conformance.precision_token_based_replay()`](https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.11/pm4py.html#pm4py.conformance.precision_token_based_replay "pm4py.conformance.precision_token_based_replay")
- alignments -> [`pm4py.conformance.precision_alignments()`](https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.11/pm4py.html#pm4py.conformance.precision_alignments "pm4py.conformance.precision_alignments")
- footprints -> [`pm4py.conformance.precision_footprints()`](https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.11/pm4py.html#pm4py.conformance.precision_footprints "pm4py.conformance.precision_footprints")

**Complexity**:


**Optimizacion**:
