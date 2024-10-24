import pandas as pd

# Cargar el CSV en un DataFrame
df = pd.read_csv('/Users/sebastianloeza/Desktop/Escuela/6to/NLP/Practicas/Practica4/corpus_sebastian.csv')

# Modificar cada mensaje de la columna 'Message'
df['Message'] = df['Message'].apply(lambda x: f"$ {x} &")

# Guardar el DataFrame modificado en un nuevo CSV
df.to_csv('corp_sebastian.csv', index=False)