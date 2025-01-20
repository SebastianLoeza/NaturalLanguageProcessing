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



data_norm = pd.DataFrame(X, columns =['features'])
data_norm['section'] = y


replacements = {
    'Economia': 'Economía',
    'Tecnología': 'Ciencia y Tecnología',
    'Deporte': 'Deportes'
}


data_norm['section'] = data_norm['section'].replace(replacements)


data_norm.to_csv('Practica5/raw/raw_classifier_corpus.csv', index=False, encoding='utf-8')