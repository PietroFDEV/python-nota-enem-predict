import pandas as pd  # Importa a biblioteca Pandas
import numpy as np  # Importa a biblioteca NumPy
from sklearn.model_selection import train_test_split, GridSearchCV  # Importa funções para divisão dos dados e busca em grade de hiperparâmetros
from sklearn.metrics import mean_squared_error  # Importa função para avaliar desempenho dos modelos
from sklearn.impute import SimpleImputer  # Importa classe para lidar com valores ausentes
from sklearn.preprocessing import StandardScaler  # Importa classe para normalizar as features
from sklearn.feature_selection import SelectFromModel  # Importa classe para seleção de features com base em modelos
from xgboost import XGBRegressor  # Importa classe para treinar modelo de regressão com XGBoost
from lightgbm import LGBMRegressor  # Importa classe para treinar modelo de regressão com LightGBM

# Carregar os dados
dados = pd.read_csv('treinamento_alunos.csv')  # Carrega os dados do arquivo CSV 'treinamento_alunos.csv' em um DataFrame Pandas

# Pré-processamento dos dados
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])  # Define as features removendo a coluna do alvo
y = dados['Original_NU_NOTA_REDACAO']  # Define o alvo como a coluna 'Original_NU_NOTA_REDACAO'

# Combinação de features
X['MEDIA_NOTAS'] = (X['NU_NOTA_CN'] + X['NU_NOTA_CH'] + X['NU_NOTA_LC'] + X['NU_NOTA_MT']) / 4  # Calcula a média das notas
X['SOMA_NOTAS'] = X['NU_NOTA_CN'] + X['NU_NOTA_CH'] + X['NU_NOTA_LC'] + X['NU_NOTA_MT']  # Calcula a soma das notas
X['DIFERENCA_MT_LC'] = X['NU_NOTA_MT'] - X['NU_NOTA_LC']  # Calcula a diferença entre as notas de Matemática e Linguagens e Códigos

# Transformações numéricas
X['LOG_MEDIA_NOTAS'] = np.log1p(X['MEDIA_NOTAS'])  # Aplica logaritmo natural na média das notas
X['RAIZ_SOMA_NOTAS'] = np.sqrt(X['SOMA_NOTAS'])  # Aplica raiz quadrada na soma das notas

# Codificar variáveis categóricas
X = pd.get_dummies(X)  # Codifica as variáveis categóricas em variáveis dummy

# Imputação de dados para lidar com valores ausentes
imputer = SimpleImputer(strategy='mean')  # Cria uma instância de SimpleImputer para imputação de valores ausentes
X_imputado = imputer.fit_transform(X)  # Imputa os valores ausentes nas features

# Normalização das features
scaler = StandardScaler()  # Cria uma instância de StandardScaler para normalização das features
X_normalizado = scaler.fit_transform(X_imputado)  # Normaliza as features

# Seleção de features com XGBoost
modelo_xgb = XGBRegressor(random_state=42)  # Cria uma instância de XGBRegressor
modelo_xgb.fit(X_normalizado, y)  # Treina o modelo XGBRegressor com as features normalizadas
selecao = SelectFromModel(modelo_xgb, prefit=True)  # Cria um seletor de features baseado no modelo XGBRegressor treinado
X_selecionado = selecao.transform(X_normalizado)  # Seleciona as features mais importantes

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_selecionado, y, test_size=0.2, random_state=42)  # Divide os dados em conjuntos de treinamento e teste

# Modelos a serem testados
modelos = [
    ('XGBoost', XGBRegressor(random_state=42)),  # Define o modelo XGBoost
    ('LightGBM', LGBMRegressor(random_state=42))  # Define o modelo LightGBM
]

# Parâmetros para ajustar
parametros = {
    'n_estimators': [100, 200, 300],  # Número de estimadores para ajustar
    'learning_rate': [0.01, 0.05, 0.1],  # Taxa de aprendizado para ajustar
    'max_depth': [3, 5, 7],  # Profundidade máxima para ajustar
    'reg_alpha': [0, 0.1, 0.5],  # Regularização L1
    'reg_lambda': [0, 0.1, 0.5]  # Regularização L2
}

# Realizar ajuste de hiperparâmetros e avaliar os modelos
for nome_modelo, modelo in modelos:  # Loop sobre os modelos a serem testados
    grid_search = GridSearchCV(modelo, parametros, cv=5, scoring='neg_mean_squared_error')  # Realiza busca em grade de hiperparâmetros
    grid_search.fit(X_treino, y_treino)  # Treina o modelo
    melhor_modelo = grid_search.best_estimator_  # Obtém o melhor modelo com os melhores hiperparâmetros
    previsoes = melhor_modelo.predict(X_teste)  # Realiza previsões no conjunto de teste
    erro = mean_squared_error(y_teste, previsoes)  # Avalia o desempenho do modelo
    print(f'Erro quadrático médio com {nome_modelo} e melhores hiperparâmetros: {erro}')  # Exibe o erro quadrático médio
