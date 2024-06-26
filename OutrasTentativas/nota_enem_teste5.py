import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Carregar os dados
dados = pd.read_csv('treinamento_alunos.csv')

# Pré-processamento dos dados
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])
y = dados['Original_NU_NOTA_REDACAO']

# Codificar variáveis categóricas
X = pd.get_dummies(X)

# Imputação de dados para lidar com valores ausentes
imputer = SimpleImputer(strategy='mean')
X_imputado = imputer.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_imputado, y, test_size=0.2, random_state=42)

# Modelos a serem testados
modelos = [
    ('XGBoost', XGBRegressor(random_state=42)),
    ('LightGBM', LGBMRegressor(random_state=42))
]

# Parâmetros para ajustar
parametros = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# Realizar ajuste de hiperparâmetros e avaliar os modelos
for nome_modelo, modelo in modelos:
    grid_search = GridSearchCV(modelo, parametros, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_treino, y_treino)
    
    # Melhor modelo com os melhores hiperparâmetros
    melhor_modelo = grid_search.best_estimator_
    
    # Previsões
    previsoes = melhor_modelo.predict(X_teste)
    
    # Avaliação do modelo com os melhores hiperparâmetros
    erro = mean_squared_error(y_teste, previsoes)
    print(f'Erro quadrático médio com {nome_modelo} e melhores hiperparâmetros: {erro}')
