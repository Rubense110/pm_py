## Virtual environment usage

Para ejecutar el proyeto, se recomienda crear un entorno virtual, para ello ejecutar este comando en terminal.
(Nota: python3 en Linux)

```
python -m venv .venv
```

Una vez hecho se habrá creado un directorio (.venv/) con el entorno virtual, para activarlo:

```
source .venv/bin/activate
```

Una vez activado podemos instalar las dependencias del proyecto simplemente ejecutando este comando:

```
pip install -r requirements.txt
```

Y finalmente, para desactivarlo:

```
deactivate
```

Para lanzar el servidor ejecutar:

```
python pm_site/manage.py runserver
```

## Funcionamiento
El proyecto se estrutura en estos ficheros:

- **problem.py** : Define el problema, como se crean y evalúan las soluciones, etc.
- **metrics.py** : Contiene las implementaciones de las métricas realizadas, son los objetivos a optimizar.
- **parameters.py** : Implementa las distintas configuraciones de hiperparámetros de los algoritmos de minería a optimizar (las variables de las soluciones).
- **utils.py** : Incorpora funciones para calcular el frente de pareto y representarlo.
- **optimize.py** : Se encarga del proceso de optimización (a partir de problem.py).
- **process_miner.py** : Interfaz de alto nivel para trabajar con la herramienta.

## Sobre las métricas
Se recomienda utilizar las métricas `basic` o `basic_conformance`. Esta última aumenta bastante los tiempos de ejecución.

## Sobre los Optimizadores
Se pueden utilizar `NSGAII`, `NSGAIII` y `SPEA2` con los hiperparámetros que se consideren.

## Sobre los mineros
Se pueden utilizar los mineros `Heurístico` e `Inductivo`

## Funcionamiento
Para ejecutar el proyecto, ejecutar el fichero `process_miner.py`. Al final del mismo se pueden cambiar los parámetros según se considere. 

Al terminarse una ejecución, en la carpeta `out` aparecerá un nuevo directorio con los resultados de la ejecución.