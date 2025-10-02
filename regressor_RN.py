import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

ARQUIVO_DESENVOLVIMENTO = './VictSim3/datasets/vict/100v/data.csv'
ARQUIVO_PREDICAO_FINAL = './VictSim3/datasets/vict/1000v/data.csv'
COLUNA_ALVO = 'sobr'
COLUNAS_FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']

df_dev = pd.read_csv(ARQUIVO_DESENVOLVIMENTO)
X_dev_full, y_dev_full = df_dev[COLUNAS_FEATURES].values, df_dev[COLUNA_ALVO].values
X_train_val, X_test_local, y_train_val, y_test_local = train_test_split(
    X_dev_full, y_dev_full, test_size=0.25, shuffle=True, random_state=42
)

parametros = {
    'num_params': 3,
    'n_layers': [2, 20, 4],
    'n_neurons': [64, 20, 128],
    'learning_rates': [0.02, 0.02, 0.001],
    'solver': 'sgd',
    'max_iter': 10000,
    'activation': 'tanh',
    'momentum': 0.95,
    'k_folds': 4
}

resultados = {
    'best_model': [],
    'model': [],
    'train_scores': [],
    'vld_scores': [],
    'best_index': [],
    'mse': [],
    'scores_ajustados': []
}

for i in range(parametros['num_params']):
  neurons_per_layer = tuple([parametros['n_neurons'][i]] * parametros['n_layers'][i])

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

  cv_results = cross_validate(
      rn,
      X_train_val,
      y_train_val,
      cv=parametros['k_folds'],
      scoring='neg_mean_squared_error',
      return_train_score=True,
      return_estimator=True
  )

  resultados['train_scores'].append(cv_results['train_score'])
  resultados['vld_scores'].append(cv_results['test_score'])
  bias = np.abs(resultados['train_scores'][i].mean() - resultados['vld_scores'][i].mean())
  
  # score ajustado penaliza o overfitting
  PESO_PENALIDADE_OVERFITTING = 1.5
  score_ajustado = resultados['vld_scores'][i].mean() - (PESO_PENALIDADE_OVERFITTING * bias)
  resultados['scores_ajustados'].append(score_ajustado)

  resultados['best_index'].append(np.argmax(resultados['vld_scores'][i]))
  resultados['best_model'].append(cv_results['estimator'][resultados['best_index'][i]])
  resultados['model'].append(cv_results['estimator'])

# resultados
print("Train & Valid Scores (Neg MSE) per parametrization:")
for i in range(parametros['num_params']):
    print(f"\nPar{i+1}\tMean\t\tVar.\t\tScores per fold")
    # ddof=1 variancia amostral
    print(f"Trn:\t{resultados['train_scores'][i].mean():>8.6f}\t{resultados['train_scores'][i].var(ddof=1):>8.6f}\t{resultados['train_scores'][i]}")
    print(f"Vld:\t{resultados['vld_scores'][i].mean():>8.6f}\t{resultados['vld_scores'][i].var(ddof=1):>8.6f}\t{resultados['vld_scores'][i]}")
    print(f"Dif:\t{np.abs(resultados['train_scores'][i].mean() - resultados['vld_scores'][i].mean()):>8.6f}\t\t\t{abs(resultados['train_scores'][i] - resultados['vld_scores'][i])}")
    print(f"Best index: {resultados['best_index'][i]}")
    print(f"Score ajustado: {resultados['scores_ajustados'][i]}")

plt.figure(figsize=(10, 6))
colors = [
    ["darkblue", "lightblue"],
    ["darkorange", "orange"],
    ["darkgreen", "lightgreen"],
]

for i in range(parametros['num_params']):
    plt.plot(range(1, len(resultados['train_scores'][i]) + 1), resultados['train_scores'][i], label=f"{i+1} Train Neg MSE", marker='o', color=colors[i][0])
    plt.plot(range(1, len(resultados['vld_scores'][i]) + 1), resultados['vld_scores'][i], label=f"{i+1} Valid Neg MSE", marker='o', color=colors[i][1])

plt.xlabel("Fold")
plt.ylabel("Neg MSE")
plt.title("Training and Validation Scores (Bias and Variance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(np.arange(1, parametros['k_folds'] + 1, 1))
plt.grid()
plt.tight_layout()
plt.show()

# escolhe o melhor parametro
melhor_param_idx = np.argmax(resultados['scores_ajustados'])
print(f"\nmelhor parametro: P{melhor_param_idx+1}")

# retreinar
print("\nretreinando modelo com todo o dataset de desenvolvimento...")
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

print("\nrealizando teste final com dataset de 1000v...")
df_teste_final = pd.read_csv(ARQUIVO_PREDICAO_FINAL)
X_teste_final = df_teste_final[COLUNAS_FEATURES].values
y_teste_verdadeiro = df_teste_final[COLUNA_ALVO].values
y_pred_final = modelo_final.predict(X_teste_final)

mse_final = mean_squared_error(y_teste_verdadeiro, y_pred_final)
print(f"\nResumo dos resultados:")
print(f"- Melhor parametro: P{melhor_param_idx+1}")
print(f"- Parametrização: {parametros['n_layers'][melhor_param_idx]} camadas, {parametros['n_neurons'][melhor_param_idx]} neurônios, lr={parametros['learning_rates'][melhor_param_idx]}")
print(f"MSE negativo final: {-mse_final:.6f}")