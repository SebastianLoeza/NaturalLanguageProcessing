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

def normalize_text(text):
    text = str(text)
    doc = nlp(text)
    # Lematizar y asegurarse de no incluir cadenas vacías
    lemmatized_tokens = [token.lemma_ for token in doc if token.lemma_.strip()]
    return ' '.join(lemmatized_tokens)

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


data_norm.to_csv('Practica5/lemm_tc/lemma_tc_classifier_corpus.csv', index=False, encoding='utf-8')