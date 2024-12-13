import pandas as pd

# Cargar el archivo CSV
ruta_csv = 'src/csv_completo.csv'  # Reemplaza con la ruta de tu archivo CSV
df = pd.read_csv(ruta_csv)

# Dividir el DataFrame según los valores de la columna 'log_name'
valor1, valor2 = df['log_name'].unique()  # Obtiene los dos valores posibles
df_valor1 = df[df['log_name'] == valor1]
df_valor2 = df[df['log_name'] == valor2]

# Guardar los DataFrames en archivos separados
df_valor1.to_csv(f'{valor1}_data.csv', index=False)
df_valor2.to_csv(f'{valor2}_data.csv', index=False)

print(f"Archivos {valor1}_data.csv y {valor2}_data.csv generados con éxito.")
