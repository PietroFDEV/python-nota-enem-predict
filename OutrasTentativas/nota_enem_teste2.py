import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Carregar os dados
dados = pd.read_csv('treinamento_alunos.csv')

# Pré-processamento dos dados
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])
y = dados['Original_NU_NOTA_REDACAO']
X = pd.get_dummies(X)  # Codificar variáveis categóricas

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir os hiperparâmetros a serem ajustados
param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Criar um objeto DecisionTreeRegressor
modelo = DecisionTreeRegressor(random_state=42)

# Criar um objeto GridSearchCV
grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='neg_mean_squared_error')

# Executar a busca de grade
grid_search.fit(X_treino, y_treino)

# Melhor modelo com os melhores hiperparâmetros
melhor_modelo = grid_search.best_estimator_

# Previsões
previsoes = melhor_modelo.predict(X_teste)

# Avaliação do modelo com os melhores hiperparâmetros
erro = mean_squared_error(y_teste, previsoes)
print(f'Erro quadrático médio com melhores hiperparâmetros: {erro}')