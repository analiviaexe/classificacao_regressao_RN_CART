import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

NOME_ARQUIVO_TREINO = './VictSim3/datasets/vict/100v/data.csv'
NOME_ARQUIVO_PREDICAO = './VictSim3/datasets/vict/1000v/data.csv'
COLUNA_ALVO = 'tri'
COLUNAS_FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim', 'gcs', 'avpu']
