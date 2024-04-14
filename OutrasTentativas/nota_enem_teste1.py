import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Carregar os dados
dados = pd.read_csv('treinamento_alunos.csv')

# Pré-processamento dos dados
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])
y = dados['Original_NU_NOTA_REDACAO']
X = pd.get_dummies(X)  # Codificar variáveis categóricas

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
modelo = DecisionTreeRegressor(random_state=42)
modelo.fit(X_treino, y_treino)

# Previsões
previsoes = modelo.predict(X_teste)

# Avaliação do modelo
erro = mean_squared_error(y_teste, previsoes)
print(f'Erro quadrático médio: {erro}')