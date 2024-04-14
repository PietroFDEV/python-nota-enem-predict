import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.metrics import mean_squared_error  
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import StandardScaler  
from sklearn.feature_selection import SelectFromModel  
from xgboost import XGBRegressor  
from lightgbm import LGBMRegressor  

# Carregar os dados
dados = pd.read_csv('treinamento_alunos.csv')  

# Pré-processamento dos dados
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])  
y = dados['Original_NU_NOTA_REDACAO']  

# Combinação de features
X['MEDIA_NOTAS'] = (X['NU_NOTA_CN'] + X['NU_NOTA_CH'] + X['NU_NOTA_LC'] + X['NU_NOTA_MT']) / 4
X['SOMA_NOTAS'] = X['NU_NOTA_CN'] + X['NU_NOTA_CH'] + X['NU_NOTA_LC'] + X['NU_NOTA_MT']
X['DIFERENCA_MT_LC'] = X['NU_NOTA_MT'] - X['NU_NOTA_LC']

# Transformações numéricas
X['LOG_MEDIA_NOTAS'] = np.log1p(X['MEDIA_NOTAS'])
X['RAIZ_SOMA_NOTAS'] = np.sqrt(X['SOMA_NOTAS'])

# Codificar variáveis categóricas
X = pd.get_dummies(X)  

# Imputação de dados para lidar com valores ausentes
imputer = SimpleImputer(strategy='mean')  
X_imputado = imputer.fit_transform(X)  

# Normalização das features
scaler = StandardScaler()  
X_normalizado = scaler.fit_transform(X_imputado)  

# Seleção de features com XGBoost
modelo_xgb = XGBRegressor(random_state=42)  
modelo_xgb.fit(X_normalizado, y)  
selecao = SelectFromModel(modelo_xgb, prefit=True)  
X_selecionado = selecao.transform(X_normalizado)  

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_selecionado, y, test_size=0.2, random_state=42)  

# Modelos a serem testados
modelos = [
    ('XGBoost', XGBRegressor(random_state=42)),  
    ('LightGBM', LGBMRegressor(random_state=42))  
]

# Parâmetros para ajustar
parametros = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Realizar ajuste de hiperparâmetros e avaliar os modelos
for nome_modelo, modelo in modelos:  
    grid_search = GridSearchCV(modelo, parametros, cv=5, scoring='neg_mean_squared_error')  
    grid_search.fit(X_treino, y_treino)  
    melhor_modelo = grid_search.best_estimator_  
    previsoes = melhor_modelo.predict(X_teste)  
    erro = mean_squared_error(y_teste, previsoes)  
    print(f'Erro quadrático médio com {nome_modelo} e melhores hiperparâmetros: {erro}')
