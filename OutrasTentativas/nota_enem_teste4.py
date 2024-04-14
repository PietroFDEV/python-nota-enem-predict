import pandas as pd 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Carregar os dados
dados = pd.read_csv('treinamento_alunos.csv')

# Pré-processamento dos dados
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])
y = dados['Original_NU_NOTA_REDACAO']
X = pd.get_dummies(X)  # Codificar variáveis categóricas

# Imputação de dados para lidar com valores ausentes
imputer = SimpleImputer(strategy='mean')
X_imputado = imputer.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_imputado, y, test_size=0.2, random_state=42)

# Instanciar e treinar o modelo de Gradient Boosting
modelo = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
modelo.fit(X_treino, y_treino)

# Fazer previsões
previsoes = modelo.predict(X_teste)

# Avaliar o modelo
erro = mean_squared_error(y_teste, previsoes)
print(f'Erro quadrático médio: {erro}')
