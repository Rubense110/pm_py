import os

## Paths
ABSPATH = os.path.abspath('.')

LOGS_PATH = os.path.join(ABSPATH, 'event_logs')
CSV_PATH = os.path.join(ABSPATH, 'csv_data')

def generate_logs_dict(logs_path=LOGS_PATH, verbose = False):
    """
    Genera un diccionario de logs de eventos basado en la estructura de directorios.
    
    Args:
        logs_path (str): Ruta al directorio que contiene los subdirectorios de logs.
    
    Returns:
        dict: Diccionario con nombres de carpetas como claves y rutas de archivos XES como valores.
    """
    logs = {}

    for folder_name in os.listdir(logs_path):
        folder_path = os.path.join(logs_path, folder_name)
        if os.path.isdir(folder_path):
            xes_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xes')]
            if xes_files:
                logs[folder_name] = os.path.join(folder_path, xes_files[0])

    if verbose:
        print("Logs encontrados:")
        for key, value in logs.items():
            print(f"{key}: {value}")

    return logs

def generate_csv_dict(csv_path=CSV_PATH, verbose = False):
   """Generate a dictionary of CSV files"""

   csvs = {}
   
   for filename in os.listdir(csv_path):
       if filename.lower().endswith('.csv'):
           key = os.path.splitext(filename)[0]
           full_path = os.path.join(csv_path, filename)
           csvs[key] = full_path
           
   if verbose:
        print("Archivos CSV encontrados:")
        for key, value in csvs.items():
            print(f"{key}: {value}")

   return csvs


logs = generate_logs_dict()

log_closed      = ('Closed', logs['Closed'])
log_financial   = ('Financial', logs['Financial'])
log_incidents   = ('Incidents', logs['Incidents'])
log_open        = ('Open', logs['Open'])

###############################################################

if __name__ == "__main__":

    LOGS = generate_logs_dict(LOGS_PATH, verbose=True)
    CSVS = generate_csv_dict(CSV_PATH, verbose= True)