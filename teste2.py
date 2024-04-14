import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import joblib

# Carregar os dados de teste
dados_teste = pd.read_csv('teste_alunos.csv')

# Preprocessamento dos dados de teste
# Imputação de dados numéricos para lidar com valores ausentes
numeric_imputer = SimpleImputer(strategy='mean')
X_numeric = dados_teste.select_dtypes(include=['float64', 'int64'])
X_numeric_imputed = numeric_imputer.fit_transform(X_numeric)

# Normalização das features numéricas
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric_imputed)

# Imputação de dados categóricos para lidar com valores ausentes
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical = dados_teste.select_dtypes(include=['object'])
X_categorical_imputed = categorical_imputer.fit_transform(X_categorical)

# Codificar variáveis categóricas
X_categorical_encoded = pd.get_dummies(pd.DataFrame(X_categorical_imputed, columns=X_categorical.columns))

# Combinação das features numéricas e categóricas
X_processed = np.concatenate((X_numeric_scaled, X_categorical_encoded), axis=1)

# Carregar o modelo treinado
lasso_model = joblib.load('lasso_model.pkl')

# Previsões com o modelo treinado
previsoes_teste = lasso_model.predict(X_processed)

# Adicionar as previsões como uma nova coluna no DataFrame de teste
dados_teste['Original_NU_NOTA_REDACAO'] = previsoes_teste

# Salvar o DataFrame com as previsões em um novo arquivo CSV
dados_teste.to_csv('teste_alunos_com_previsoes.csv', index=False)
