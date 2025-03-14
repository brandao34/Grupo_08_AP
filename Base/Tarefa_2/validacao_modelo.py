import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# Agora importe a classe NeuralNetwork
from models.nn import NeuralNetwork

# Carregar modelo e vetorizador
with open('modelo_nn.pkl', 'rb') as f:
    nn_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Carregar datasets de input/output
df_input = pd.read_csv('dataset1_inputs.csv', sep='\t')
df_output = pd.read_csv('dataset1_outputs.csv', sep='\t')

# Função de pré-processamento compatível
def preprocessing(df, vectorizer):
    X = vectorizer.transform(df['Text']).toarray()  # Note o 'Text' maiúsculo
    return X

# Converter textos para features
X_new = preprocessing(df_input, tfidf)

# Fazer previsões
predictions = nn_model.predict(X_new)
df_output['Predicted'] = np.where(predictions >= 0.5, 'AI', 'Human')

# Mapeamento de labels
label_map = {'Human': 0, 'AI': 1}
df_output['Label_num'] = df_output['Label'].map(label_map)
df_output['Predicted_num'] = df_output['Predicted'].map(label_map)

# Matriz de confusão
cm = confusion_matrix(df_output['Label_num'], df_output['Predicted_num'])

# Métricas
print("\nRelatório de Classificação:")
print(classification_report(df_output['Label'], df_output['Predicted']))

# Acurácia por classe
print("\Accuracy por Classe:")
class_report = classification_report(df_output['Label'], df_output['Predicted'], output_dict=True)
for cls in ['Human', 'AI']:
    print(f"{cls}: {class_report[cls]['precision']*100:.2f}% de precisão")