import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
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
    'num_params': 3,            # num de parametrizacoes a treinar
    'n_layers': [2, 20, 4],
    'n_neurons': [64, 20, 128],
    'learning_rates': [0.02, 0.02, 0.001],
    'solver': 'sgd',
    'max_iter': 10000,
    'activation': 'tanh',
    'momentum': 0.95,
    'k_folds': 4
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

plt.figure(figsize=(10, 6))
colors = [
    ["darkblue", "lightblue"],  # Tons de azul para i=0
    ["darkorange", "orange"],  # Tons de laranja para i=1
    ["darkgreen", "lightgreen"],  # Tons de verde para i=2 (se houver mais índices)
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

print("\n5.3) Escolhendo a melhor parametrização (Método: Penalidade por Overfitting)...")

# Fator de penalização: quão importante é a diferença?
# Se 1.0, uma diferença de 0.1 tem o mesmo peso que uma queda de 0.1 na média.
PESO_PENALIDADE_OVERFITTING = 1.5

scores_ajustados = []
for i in range(parametros['num_params']):
    # Pega as médias dos scores
    media_validacao = resultados['vld_scores'][i].mean()
    media_treino = resultados['train_scores'][i].mean()

    # Calcula a diferença (gap) que pode indicar overfitting
    diferenca_overfitting = abs(media_treino - media_validacao)

    # O score ajustado penaliza o overfitting
    score_ajustado = media_validacao - (PESO_PENALIDADE_OVERFITTING * diferenca_overfitting)
    scores_ajustados.append(score_ajustado)

    print(f"Par{i+1}: Média Vld={media_validacao:.6f}, Diferença={diferenca_overfitting:.6f} -> Score Ajustado={score_ajustado:.6f}")

# Escolhe o índice da parametrização com o MAIOR score ajustado
melhor_param_idx = np.argmax(scores_ajustados)

print(f"\nMelhor parametrização escolhida: P{melhor_param_idx+1} (melhor balanço entre performance e generalização)")

# Retreinar
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

# Teste final com dataset de 1000v
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