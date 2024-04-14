import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

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

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)

# Lasso Regression with Polynomial Features
degree = 2
alpha = 0.1
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X_treino)
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_poly, y_treino)
X_test_poly = poly_features.transform(X_teste)
previsoes_lasso = lasso_model.predict(X_test_poly)
erro_lasso = mean_squared_error(y_teste, previsoes_lasso)
print(f'Erro quadrático médio com Lasso Regression: {erro_lasso}')
