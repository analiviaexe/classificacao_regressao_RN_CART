import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ARQUIVO_DESENVOLVIMENTO = './VictSim3/datasets/vict/100v/data.csv'
ARQUIVO_PREDICAO_FINAL = './VictSim3/datasets/vict/1000v/data.csv'
COLUNA_ALVO = 'tri'
COLUNAS_FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim', 'gcs', 'avpu']

df_dev = pd.read_csv(ARQUIVO_DESENVOLVIMENTO)
X_dev_full, y_dev_full = df_dev[COLUNAS_FEATURES].values, df_dev[COLUNA_ALVO].values
X_train_val, X_test_local, y_train_val, y_test_local = train_test_split(
    X_dev_full, y_dev_full, test_size=0.25, shuffle=True, random_state=42
)

modelo_base = DecisionTreeClassifier(random_state=42)
parametros_lista = [
    { 'criterion': ['entropy'], 'max_depth': [4], 'min_samples_leaf': [8]},
    { 'criterion': ['entropy'], 'max_depth': [8], 'min_samples_leaf': [12]},
    { 'criterion': ['entropy'], 'max_depth': [8], 'min_samples_leaf': [4]}
]

grid_search = GridSearchCV(
    estimator=modelo_base,
    param_grid=parametros_lista,
    scoring='f1_weighted',
    cv=4,
    return_train_score=True,
    verbose=4
)
grid_search.fit(X_train_val, y_train_val)
resultados = grid_search.cv_results_
nomes_p = [f"P{i+1}" for i in range(len(resultados['params']))]

# medias dos parametros
print("\n4) Comparar as médias dos f-scores")
header4 = "| {:<14} |".format("Média f-score")
for nome in nomes_p: header4 += " {:<5} |".format(nome)
print(header4); print("-" * len(header4))
linha_treino4 = "| {:<14} |".format("Treino")
linha_valid4 = "| {:<14} |".format("Validação")
for score in resultados['mean_train_score']: linha_treino4 += " {:.4f} |".format(score)
for score in resultados['mean_test_score']: linha_valid4 += " {:.4f} |".format(score)
print(linha_treino4); print(linha_valid4)

# variancias dos parametros
print("\n5) Comparar as variâncias dos f-scores")
header5 = "| {:<14} |".format("Variância f-score")
for nome in nomes_p: header5 += " {:<5} |".format(nome)
print(header5); print("-" * len(header5))
linha_treino5 = "| {:<14} |".format("Treino")
linha_valid5 = "| {:<14} |".format("Validação")
for std in resultados['std_train_score']: linha_treino5 += " {:.4f} |".format(std**2)
for std in resultados['std_test_score']: linha_valid5 += " {:.4f} |".format(std**2)
print(linha_treino5); print(linha_valid5)

print(f"\nMelhores parâmetros encontrados: {grid_search.best_estimator_}")
modelo_final = grid_search.best_estimator_

# --- predições  ---
# com dados do treinamento
y_pred_train = modelo_final.predict(X_train_val)
acc_train = accuracy_score(y_train_val, y_pred_train) * 100
print(f"Acuracia com dados de treino: {acc_train:.2f}%")

# com dados de teste
y_pred_test = modelo_final.predict(X_test_local)
acc_test = accuracy_score(y_test_local, y_pred_test) * 100
print(f"Acuracia com dados de teste: {acc_test:.2f}%")

# retreinando modelo com toda a amostra
melhores_parametros = grid_search.best_params_
modelo_final = DecisionTreeClassifier(random_state=42, **melhores_parametros)
modelo_final.fit(X_dev_full, y_dev_full)

# fazendo predição com modelo de 1000 vitimas
df_teste_final = pd.read_csv(ARQUIVO_PREDICAO_FINAL)
X_teste_final = df_teste_final[COLUNAS_FEATURES].values
y_teste_verdadeiro = df_teste_final[COLUNA_ALVO].values
y_pred_final = modelo_final.predict(X_teste_final)

print("\nmatriz de confusao:")
print(classification_report(y_teste_verdadeiro, y_pred_final))
ConfusionMatrixDisplay.from_predictions(y_teste_verdadeiro, y_pred_final)
plt.show()