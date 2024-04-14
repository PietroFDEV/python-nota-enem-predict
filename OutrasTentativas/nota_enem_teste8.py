import pandas as pd  
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

# Parâmetros para ajustar
parametros = {
    'n_estimators': [100, 200, 300], 
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Realizar ajuste de hiperparâmetros e avaliar o modelo XGBoost
grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), parametros, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_treino, y_treino)

# Melhor modelo com os melhores hiperparâmetros
melhor_modelo_xgb = grid_search_xgb.best_estimator_
previsoes_xgb = melhor_modelo_xgb.predict(X_teste)
erro_xgb = mean_squared_error(y_teste, previsoes_xgb)
print(f'Erro quadrático médio com XGBoost e melhores hiperparâmetros: {erro_xgb}')

# Realizar ajuste de hiperparâmetros e avaliar o modelo LightGBM
grid_search_lgbm = GridSearchCV(LGBMRegressor(random_state=42), parametros, cv=5, scoring='neg_mean_squared_error')
grid_search_lgbm.fit(X_treino, y_treino)

# Melhor modelo com os melhores hiperparâmetros
melhor_modelo_lgbm = grid_search_lgbm.best_estimator_
previsoes_lgbm = melhor_modelo_lgbm.predict(X_teste)
erro_lgbm = mean_squared_error(y_teste, previsoes_lgbm)
print(f'Erro quadrático médio com LightGBM e melhores hiperparâmetros: {erro_lgbm}')
