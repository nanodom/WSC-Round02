import pandas as pd

# 1. Cargar la matriz original
demand_df = pd.read_csv('demand_matrix.csv')
origins = demand_df.iloc[:, 0]
numeric_cols = demand_df.columns[1:]

# 2. Aplicar el factor de estrés (ej. multiplicar por 1.0) y CONVERTIR A ENTEROS
demand_df[numeric_cols] = (demand_df[numeric_cols] * 1.0).round().astype(int)

# 3. Guardar el archivo corregido
demand_df.to_csv('demand_matrix.csv', index=False) # O el nombre de tu archivo de salida
print("¡Matriz de demanda estresada corregida a formato entero con éxito!")