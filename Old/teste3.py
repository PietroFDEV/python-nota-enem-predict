import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import joblib

# Load the training data
dados = pd.read_csv('treinamento_alunos.csv')

# Preprocessing of the training data
X = dados.drop(columns=['Original_NU_NOTA_REDACAO'])
y = dados['Original_NU_NOTA_REDACAO']

# Feature engineering
X['MEDIA_NOTAS'] = (X['NU_NOTA_CN'] + X['NU_NOTA_CH'] + X['NU_NOTA_LC'] + X['NU_NOTA_MT']) / 4
X['SOMA_NOTAS'] = X['NU_NOTA_CN'] + X['NU_NOTA_CH'] + X['NU_NOTA_LC'] + X['NU_NOTA_MT']
X['DIFERENCA_MT_LC'] = X['NU_NOTA_MT'] - X['NU_NOTA_LC']

# Numeric transformations
X['LOG_MEDIA_NOTAS'] = np.log1p(X['MEDIA_NOTAS'])
X['RAIZ_SOMA_NOTAS'] = np.sqrt(X['SOMA_NOTAS'])

# Encode categorical variables
X = pd.get_dummies(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Lasso Regression with Polynomial Features
degree = 2
alpha = 0.1
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X_train)
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_poly, y_train)
X_test_poly = poly_features.transform(X_test)
previsoes_lasso = lasso_model.predict(X_test_poly)
erro_lasso = mean_squared_error(y_test, previsoes_lasso)
print(f'Erro quadrático médio com Lasso Regression: {erro_lasso}')

# Save the trained model to a pickle file
joblib.dump(lasso_model, 'lasso_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the test data
dados_teste = pd.read_csv('teste_alunos.csv')

# Preprocessing of the test data
if 'Original_NU_NOTA_REDACAO' in dados_teste.columns:
    dados_teste.drop(columns=['Original_NU_NOTA_REDACAO'], inplace=True)

dados_teste['MEDIA_NOTAS'] = (dados_teste['NU_NOTA_CN'] + dados_teste['NU_NOTA_CH'] + dados_teste['NU_NOTA_LC'] + dados_teste['NU_NOTA_MT']) / 4
dados_teste['SOMA_NOTAS'] = dados_teste['NU_NOTA_CN'] + dados_teste['NU_NOTA_CH'] + dados_teste['NU_NOTA_LC'] + dados_teste['NU_NOTA_MT']
dados_teste['DIFERENCA_MT_LC'] = dados_teste['NU_NOTA_MT'] - dados_teste['NU_NOTA_LC']

dados_teste['LOG_MEDIA_NOTAS'] = np.log1p(dados_teste['MEDIA_NOTAS'])
dados_teste['RAIZ_SOMA_NOTAS'] = np.sqrt(dados_teste['SOMA_NOTAS'])

dados_teste = pd.get_dummies(dados_teste)

# Load the trained model and preprocessing transformers
lasso_model = joblib.load('lasso_model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Apply transformations to the test data
dados_teste_imputed = imputer.transform(dados_teste)
dados_teste_scaled = scaler.transform(dados_teste_imputed)

# Make predictions on the test data
previsoes_teste = lasso_model.predict(dados_teste_scaled)

# Save the predictions to a file or use as needed
