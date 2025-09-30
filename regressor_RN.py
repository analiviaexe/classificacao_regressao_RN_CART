import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

ARQUIVO_DESENVOLVIMENTO = './VictSim3/datasets/vict/100v/data.csv'
ARQUIVO_PREDICAO_FINAL = './VictSim3/datasets/vict/1000v/data.csv'
COLUNA_ALVO = 'tri'
COLUNAS_FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim', 'gcs', 'avpu']

df_dev = pd.read_csv(ARQUIVO_DESENVOLVIMENTO)
X_dev_full, y_dev_full = df_dev[COLUNAS_FEATURES].values, df_dev[COLUNA_ALVO].values
X_train_val, X_test_local, y_train_val, y_test_local = train_test_split(
    X_dev_full, y_dev_full, test_size=0.25, shuffle=True, random_state=42
)

parametros = {
    'num_params': 3,            # num de parametrizacoes a treinar
    'n_layers': [2, 6, 20],
    'n_neurons': [4, 10, 20],
    'learning_rates': [0.01, 0.025, 0.03],
    'solver': 'sgd',
    'max_iter': 10000,
    'activation': 'tanh',
    'momentum': 0.95,
    'k_folds': 5
}

results = []

resultados = {
    'best_model': [],  # armazenar o melhor modelo de cada parametrização
    'model': [],        # todos os modelos de cada parametrização
    'train_scores': [],
    'vld_scores': [],
    'best_index': [],
    'mse': []
}

for i in range(parametros['num_params']):
  # Correctly create a tuple for hidden_layer_sizes
  neurons_per_layer = tuple([parametros['n_neurons'][i]] * parametros['n_layers'][i])

  print(f"training MLP with hidden layers: {neurons_per_layer}")
  rn = MLPRegressor(
      hidden_layer_sizes=neurons_per_layer,
      activation=parametros['activation'],
      solver=parametros['solver'],
      learning_rate='adaptive',
      learning_rate_init=parametros['learning_rates'][i],
      max_iter=parametros['max_iter'],
      shuffle=True,
      momentum=parametros['momentum'],
      random_state=42
      )

  # cross_validate faz fit (treinamento) e retorna um modelo aprendido por fold,
  # o score de treino e o de validação (ver mais abaixo)
  cv_results = cross_validate(
      rn,
      X_train_val,
      y_train_val,
      cv=parametros['k_folds'],
      scoring='neg_mean_squared_error', # scikit usa negativo para manter o padrao de quanto mais alto o score, melhor o estimador
                                        # Neste caso, quanto maior o MSE (negativo), melhor será porque mede o erro.
      return_train_score=True,  # Include training scores
      return_estimator=True     # Include trained models
  )

  # MSE de treino e validação (erro quadrático médio, mean squared error)
  resultados['train_scores'].append(cv_results['train_score'])
  resultados['vld_scores'].append(cv_results['test_score'])
  print(f"NEG MSE treino   : {resultados['train_scores'][i]}")
  print(f"NEG MSE validação: {resultados['vld_scores'][i]}\n")

  # Diferença absoluta do MSE do treinamento e de validação
  bias = np.abs(resultados['train_scores'][i] - resultados['vld_scores'][i])

  # Salva o indice do menor score)
  resultados['best_index'].append(np.argmax(resultados['vld_scores'][i]))

  # Salva o modelo que apresenta melhor score (MSE)
  resultados['best_model'].append(cv_results['estimator'][resultados['best_index'][i]])

  # Salva todos os modelos da parametrizacao
  resultados['model'].append(cv_results['estimator'])

  #print(f"Parametrization {i+1}: {model}")
  print(f"Best Index: {resultados['best_index'][i]}\n")

# Resultados
print("Train & Valid Scores (Neg MSE) per parametrization:")
for i in range(parametros['num_params']):
    print(f"Par{i+1}\tMean\t\tVar.\t\tScores per fold")
    # ddof=1 variancia amostral
    print(f"Trn:\t{resultados['train_scores'][i].mean():>8.6f}\t{resultados['train_scores'][i].var(ddof=1):>8.6f}\t{resultados['train_scores'][i]}")
    print(f"Vld:\t{resultados['vld_scores'][i].mean():>8.6f}\t{resultados['vld_scores'][i].var(ddof=1):>8.6f}\t{resultados['vld_scores'][i]}")
    print(f"Dif:\t{np.abs(resultados['train_scores'][i].mean() - resultados['vld_scores'][i].mean()):>8.6f}\t\t\t{abs(resultados['train_scores'][i] - resultados['vld_scores'][i])}")
    print(f"Best index: {resultados['best_index'][i]}")
    print()