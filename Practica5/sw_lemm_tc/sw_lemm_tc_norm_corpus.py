import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy

# Load data and models
data = pd.read_csv('Practica5/raw_corpuses_final.csv')
data.drop_duplicates(subset='Title', inplace= True)

data['features'] = data['Title'].astype(str) + ' ' + data['Content'].astype(str)  # Convert to string
data_filtered = data[['features', 'Section']]

X = data_filtered['features'].values
y = data_filtered['Section'].values

nlp = spacy.load('es_core_news_sm')

# Your stopwords definitions remain the same
articulos = {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'}
preposiciones = {
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante',
    'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía'
}
conjunciones = {
    'y', 'e', 'ni', 'que', 'o', 'u', 'pero', 'mas', 'sino', 'aunque', 'si', 'como', 'porque', 'pues', 'ya', 'luego',
    'así', 'bien', 'sea', 'fuera', 'bien', 'cuando', 'donde'
}
pronombres = {
    'yo', 'tú', 'él', 'ella', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas', 'me', 'te', 'se', 'nos',
    'os', 'mi', 'mis', 'tu', 'tus', 'su', 'sus', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra',
    'vuestros', 'vuestras', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos',
    'suyas', 'le', 'les', 'lo', 'la', 'los', 'las', 'él', 'ella', 'ellos', 'ellas', 'esto', 'estos', 'esta', 'estas',
    'eso', 'esos', 'esa', 'esas', 'aquel', 'aquellos', 'aquella', 'aquellas', 'quien', 'quienes', 'cual', 'cuales',
    'cuya', 'cuyo', 'cuyas', 'cuyos'
}
signos = {
    '.',',','\'','-',':','\"',';','“','”'
}
stopWords = articulos | preposiciones | conjunciones | pronombres | signos

def normalize_text(text):
    text = str(text)
    doc = nlp(text)
    # Filtrar tokens no deseados y eliminar cadenas vacías
    filtered_tokens = [token.lemma_ for token in doc if token.text.lower() not in stopWords and token.lemma_.strip()]
    return ' '.join(filtered_tokens)

X_norm = []
for text in X:
    try:
        normalized = normalize_text(text)
        X_norm.append(normalized)
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error message: {str(e)}")
        X_norm.append("")  
X_norm = np.array(X_norm)


data_norm = pd.DataFrame(X_norm, columns =['features'])
data_norm['section'] = y


replacements = {
    'Economia': 'Economía',
    'Tecnología': 'Ciencia y Tecnología',
    'Deporte': 'Deportes'
}


data_norm['section'] = data_norm['section'].replace(replacements)


data_norm.to_csv('Practica5/sw_lemm_tc/sw_lemm_tc_classifier_corpus.csv', index=False, encoding='utf-8')