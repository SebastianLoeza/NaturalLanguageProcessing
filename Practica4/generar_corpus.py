from datetime import datetime

# Define la fecha límite a partir de la cual se quiere conservar las líneas (formato: DD/MM/YY)
fecha_limite = datetime.strptime('21/04/24', '%d/%m/%y')

# Abrir el archivo y leer todas las líneas
with open('chats/chat_6am1.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Procesar las líneas y escribir las válidas en un nuevo archivo
with open('corpus/chat_6am1.txt', 'w', encoding='utf-8') as file:
    for line in lines:
        # Obtener la fecha de la línea
        try:
            fecha_str = line.split(']')[0][1:]  # Extrae la parte de la fecha y hora
            fecha = datetime.strptime(fecha_str.split(',')[0], '%d/%m/%y')
        except (ValueError, IndexError):
            continue  # Si no puede leer la fecha, salta a la siguiente línea

        # Filtrar líneas después de la fecha límite
        if fecha >= fecha_limite:
            # Si es un mensaje de Pika, limpiar el texto
            if "Pika:" in line:
                # Extraer el texto después de "Pika:"
                texto = line.split("Pika:")[1].strip()
                file.write(texto + '\n')
