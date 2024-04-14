import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import joblib

# Carregar os dados
dados = pd.read_csv('treinamento_alunos.csv')

# Pré-processamento dos dados
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])
y = dados['Original_NU_NOTA_REDACAO']

# Imputação de dados numéricos para lidar com valores ausentes
numeric_imputer = SimpleImputer(strategy='mean')
X_numeric = X.select_dtypes(include=['float64', 'int64'])
X_numeric_imputed = numeric_imputer.fit_transform(X_numeric)

# Normalização das features numéricas
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric_imputed)

# Imputação de dados categóricos para lidar com valores ausentes
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical = X.select_dtypes(include=['object'])
X_categorical_imputed = categorical_imputer.fit_transform(X_categorical)

# Codificar variáveis categóricas
X_categorical_encoded = pd.get_dummies(pd.DataFrame(X_categorical_imputed, columns=X_categorical.columns))

# Combinação das features numéricas e categóricas
X_processed = np.concatenate((X_numeric_scaled, X_categorical_encoded), axis=1)

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_processed, y, test_size=0.2, random_state=42)

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

# Save the trained model to a pickle file
joblib.dump(lasso_model, 'lasso_model.pkl')
