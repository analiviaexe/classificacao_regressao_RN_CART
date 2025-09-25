
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

NOME_ARQUIVO_TREINO = './VictSim3/datasets/vict/100v/data.csv'
NOME_ARQUIVO_PREDICAO = './VictSim3/datasets/vict/1000v/data.csv'
COLUNA_ALVO = 'tri'
COLUNAS_FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim', 'gcs', 'avpu']


# ==============================================================================
# --- CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
# ==============================================================================
print("--- Iniciando Preparação dos Dados ---")
try:
    # Carregar dados de TREINO
    df_treino = pd.read_csv(NOME_ARQUIVO_TREINO)
    X_treino_full = df_treino[COLUNAS_FEATURES].values
    y_treino_full = df_treino[COLUNA_ALVO].values
    print(f"Dataset de treino '{NOME_ARQUIVO_TREINO}' carregado.")

    # Carregar dados para PREDIÇÃO FINAL (Passo 6.5)
    df_predicao = pd.read_csv(NOME_ARQUIVO_PREDICAO)
    X_para_predicao = df_predicao[COLUNAS_FEATURES].values
    print(f"Dataset de predição final carregado. Formato: {X_para_predicao.shape}")

except FileNotFoundError:
    print(f"ERRO: O arquivo de treino '{NOME_ARQUIVO_TREINO}' não foi encontrado!")
    # Encerra o script se o arquivo de treino não existir
    exit()
except KeyError as e:
    print(f"ERRO: Uma das colunas não foi encontrada no CSV. Verifique os nomes em COLUNA_ALVO e COLUNAS_FEATURES. Detalhe: {e}")
    exit()
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
    exit()


# ==============================================================================
# --- PASSOS 6.1 e 6.2: TREINO COM K-FOLD E COMPARAÇÃO DE PARÂMETROS ---
# ==============================================================================

def avaliar_classificador(params_dict, X, y):
    """
    Função para treinar o DecisionTreeClassifier com validação cruzada para diferentes parametrizações.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    resultados = {}
    
    print("\n--- Iniciando Avaliação com Validação Cruzada (10-fold) ---")
    
    for nome_param, params in params_dict.items():
        fscores_treino_fold, fscores_valid_fold = [], []

        for fold, (idx_treino, idx_valid) in enumerate(kf.split(X)):
            X_treino, X_valid = X[idx_treino], X[idx_valid]
            y_treino, y_valid = y[idx_treino], y[idx_valid]

            modelo = DecisionTreeClassifier(**params)
            modelo.fit(X_treino, y_treino)

            # Avalia no treino (para análise de overfitting)
            pred_treino = modelo.predict(X_treino)
            score_treino = f1_score(y_treino, pred_treino, average='weighted', zero_division=0)
            fscores_treino_fold.append(score_treino)

            # Avalia na validação (para análise de generalização)
            pred_valid = modelo.predict(X_valid)
            score_valid = f1_score(y_valid, pred_valid, average='weighted', zero_division=0)
            fscores_valid_fold.append(score_valid)

        # Consolida resultados da parametrização
        resultados[nome_param] = {
            'media_treino': np.mean(fscores_treino_fold),
            'var_treino': np.var(fscores_treino_fold),
            'media_valid': np.mean(fscores_valid_fold),
            'var_valid': np.var(fscores_valid_fold)
        }
        
        print(f"\nResultados para {nome_param} (parâmetros: {params}):")
        print(f"  F1-Score de Treino : Média = {resultados[nome_param]['media_treino']:.4f} | Variância = {resultados[nome_param]['var_treino']:.4f}")
        print(f"  F1-Score de Valid.: Média = {resultados[nome_param]['media_valid']:.4f} | Variância = {resultados[nome_param]['var_valid']:.4f}")

    # Escolhe a melhor parametrização com base na maior média do F1-Score de validação
    melhor_nome = max(resultados, key=lambda k: resultados[k]['media_valid'])
        
    print(f"\n--- Análise de Viés e Variância (Passo 6.2) ---")
    print("Compare os scores de treino e validação:")
    print("  - Se Treino >> Valid. => Overfitting (alta variância).")
    print("  - Se Treino ≈ Valid. (ambos baixos) => Underfitting (alto viés).")
    print("  - Se Treino ≈ Valid. (ambos altos) => Bom equilíbrio!")
    print(f"\nMelhor parametrização escolhida: {melhor_nome}")
    
    return melhor_nome, params_dict[melhor_nome]

# 6.1. Escolher três parametrizações diferentes para treino/validação.
params_classificador = {
    'P1_simples':  {'max_depth': 3, 'random_state': 42},
    'P2_medio':    {'max_depth': 8, 'random_state': 42},
    'P3_complexo': {'max_depth': None, 'random_state': 42}
}

# 6.2. Comparar as três parametrizações e escolher a melhor.
nome_melhor_param, melhores_parametros = avaliar_classificador(
    params_classificador, X_treino_full, y_treino_full
)


# ==============================================================================
# --- PASSOS 6.3 e 6.4: ESCOLHA DO MODELO E RETREINAMENTO ---
# ==============================================================================
print(f"\n--- Passos 6.3 e 6.4: Escolha e Retreinamento ---")

# 6.3. Critério de escolha do modelo
print("Critério de escolha: Selecionar a parametrização com o maior F1-Score médio na validação,")
print("pois representa a melhor capacidade de generalização do modelo para dados não vistos.")

# 6.4. Retreinar o modelo escolhido com todo o dataset
print(f"Retreinando um novo modelo com os parâmetros de '{nome_melhor_param}': {melhores_parametros}")
modelo_final = DecisionTreeClassifier(**melhores_parametros)
modelo_final.fit(X_treino_full, y_treino_full)


# ==============================================================================
# --- PASSO 6.5: PREDIÇÃO COM O MODELO RETREINADO ---
# ==============================================================================
print("\n--- Passo 6.5: Predição com Modelo Final ---")
predicoes_finais = modelo_final.predict(X_para_predicao)

# Adiciona as predições como uma nova coluna no DataFrame de predição para fácil visualização
df_predicao['predicao_' + COLUNA_ALVO] = predicoes_finais

print("Amostra das predições:")
# Mostra algumas colunas importantes junto com a predição para comparação
colunas_para_mostrar = COLUNAS_FEATURES[:4] + ['predicao_' + COLUNA_ALVO]
print(df_predicao[colunas_para_mostrar].head(15))

# Salva o resultado em um novo arquivo CSV (opcional)
# nome_arquivo_saida = 'predicoes.csv'
# df_predicao.to_csv(nome_arquivo_saida, index=False)
# print(f"\nResultados completos salvos em '{nome_arquivo_saida}'")