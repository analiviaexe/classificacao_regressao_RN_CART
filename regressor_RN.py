import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ARQUIVO_DESENVOLVIMENTO = './VictSim3/datasets/vict/100v/data.csv'
ARQUIVO_PREDICAO_FINAL = './VictSim3/datasets/vict/1000v/data.csv'
COLUNA_ALVO = 'tri'
COLUNAS_FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']

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

# 5.2) Comparar as médias dos MSE negativos
nomes_p = [f"P{i+1}" for i in range(parametros['num_params'])]
print("\n5.2) Comparar as médias dos MSE negativos")
header_media = "| {:<14} |".format("Média MSE Neg")
for nome in nomes_p: header_media += " {:<7} |".format(nome)
print(header_media); print("-" * len(header_media))
linha_treino_media = "| {:<14} |".format("Treino")
linha_valid_media = "| {:<14} |".format("Validação")
for i in range(parametros['num_params']): 
    linha_treino_media += " {:.5f} |".format(resultados['train_scores'][i].mean())
for i in range(parametros['num_params']): 
    linha_valid_media += " {:.5f} |".format(resultados['vld_scores'][i].mean())
print(linha_treino_media); print(linha_valid_media)

# Comparar as variâncias dos MSE negativos
print("\n5.2) Comparar as variâncias dos MSE negativos")
header_var = "| {:<14} |".format("Variância MSE Neg")
for nome in nomes_p: header_var += " {:<7} |".format(nome)
print(header_var); print("-" * len(header_var))
linha_treino_var = "| {:<14} |".format("Treino")
linha_valid_var = "| {:<14} |".format("Validação")
for i in range(parametros['num_params']): 
    linha_treino_var += " {:.5f} |".format(resultados['train_scores'][i].var(ddof=1))
for i in range(parametros['num_params']): 
    linha_valid_var += " {:.5f} |".format(resultados['vld_scores'][i].var(ddof=1))
print(linha_treino_var); print(linha_valid_var)

# 5.2) Análise de viés e variância para escolha da melhor parametrização
print("\n5.2) Análise de viés/variância e escolha da melhor parametrização:")
bias_scores = []
for i in range(parametros['num_params']):
    bias = abs(resultados['train_scores'][i].mean() - resultados['vld_scores'][i].mean())
    variance = resultados['vld_scores'][i].var(ddof=1)
    bias_scores.append((bias, variance, resultados['vld_scores'][i].mean()))
    print(f"P{i+1}: Viés={bias:.6f}, Variância={variance:.6f}, MSE_val={resultados['vld_scores'][i].mean():.6f}")

# Escolher parametrização com melhor trade-off (menor viés + menor variância + melhor MSE validação)
melhor_param_idx = max(range(len(bias_scores)), key=lambda i: bias_scores[i][2])  # Maior MSE negativo (menor erro)
print(f"\nMelhor parametrização escolhida: P{melhor_param_idx+1}")

# 5.3) Escolher o melhor modelo da parametrização escolhida
melhor_modelo = resultados['best_model'][melhor_param_idx]
print(f"5.3) Modelo escolhido: fold {resultados['best_index'][melhor_param_idx]} da parametrização P{melhor_param_idx+1}")

# 5.4) Retreinar com todo o dataset de desenvolvimento
print("\n5.4) Retreinando modelo com todo o dataset de desenvolvimento...")
neurons_per_layer = tuple([parametros['n_neurons'][melhor_param_idx]] * parametros['n_layers'][melhor_param_idx])
modelo_final = MLPRegressor(
    hidden_layer_sizes=neurons_per_layer,
    activation=parametros['activation'],
    solver=parametros['solver'],
    learning_rate='adaptive',
    learning_rate_init=parametros['learning_rates'][melhor_param_idx],
    max_iter=parametros['max_iter'],
    shuffle=True,
    momentum=parametros['momentum'],
    random_state=42
)
modelo_final.fit(X_dev_full, y_dev_full)

# 5.5) Teste final com dataset de 1000v
print("\n5.5) Realizando teste final com dataset de 1000v...")
df_teste_final = pd.read_csv(ARQUIVO_PREDICAO_FINAL)
X_teste_final = df_teste_final[COLUNAS_FEATURES].values
y_teste_verdadeiro = df_teste_final[COLUNA_ALVO].values
y_pred_final = modelo_final.predict(X_teste_final)

# Calcular MSE final
from sklearn.metrics import mean_squared_error
mse_final = mean_squared_error(y_teste_verdadeiro, y_pred_final)
print(f"MSE final no dataset de teste (1000v): {mse_final:.6f}")
print(f"MSE negativo final: {-mse_final:.6f}")

print(f"\nResumo dos resultados:")
print(f"- Melhor parametrização: P{melhor_param_idx+1}")
print(f"- Configuração: {parametros['n_layers'][melhor_param_idx]} camadas, {parametros['n_neurons'][melhor_param_idx]} neurônios, lr={parametros['learning_rates'][melhor_param_idx]}")
print(f"- MSE final: {mse_final:.6f}")

plt.figure(figsize=(8, 6))
plt.plot(X_plot, true_fun(X_plot), label='Função verdadeira', color='black', alpha=0.7)

for i in range(num_params):
  color = cm.viridis(i / (num_params - 1))
  plt.plot(X_plot, best_model[i].predict(X_plot.reshape(-1, 1)),
           label=r'$\hat{f}_{' + str(i+1) + '}(X)$', alpha=0.7)

plt.xlabel('X')
plt.ylabel('y')
plt.title('X vs y')
plt.legend()
plt.grid(True)
plt.show()