import csv

# Nombre del archivo de entrada y salida
archivo_txt = 'scripts_gen_corpus/corpus_sebastian.txt'
archivo_csv = 'corpus_sebastian.csv'

# Abrir el archivo .txt y leer el contenido
with open(archivo_txt, 'r', encoding='utf-8') as file_txt:
    lineas = file_txt.readlines()

# Escribir el archivo .csv
with open(archivo_csv, 'w', newline='', encoding='utf-8') as file_csv:
    writer = csv.writer(file_csv)

    for linea in lineas:
        # Escribir la línea como un solo elemento en el CSV, entre comillas dobles si es necesario
        writer.writerow([linea.strip()])  # Cada línea
