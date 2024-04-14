import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib

# Load the test data
test_data = pd.read_csv('teste_alunos.csv')

# Pré-processamento dos dados
X_teste = test_data.copy()

# Combinação de features
X_teste['MEDIA_NOTAS'] = (X_teste['NU_NOTA_CN'] + X_teste['NU_NOTA_CH'] + X_teste['NU_NOTA_LC'] + X_teste['NU_NOTA_MT']) / 4
X_teste['SOMA_NOTAS'] = X_teste['NU_NOTA_CN'] + X_teste['NU_NOTA_CH'] + X_teste['NU_NOTA_LC'] + X_teste['NU_NOTA_MT']
X_teste['DIFERENCA_MT_LC'] = X_teste['NU_NOTA_MT'] - X_teste['NU_NOTA_LC']

# Transformações numéricas
X_teste['LOG_MEDIA_NOTAS'] = np.log1p(X_teste['MEDIA_NOTAS'])
X_teste['RAIZ_SOMA_NOTAS'] = np.sqrt(X_teste['SOMA_NOTAS'])

# Codificar variáveis categóricas
X_teste = pd.get_dummies(X_teste)

# Load the pre-trained XGBoost model
loaded_model = joblib.load('XGBoost_model.pkl')

# Imputation for test data
imputer = SimpleImputer(strategy='mean')
test_data_imputed = imputer.transform(X_teste)

# Normalization for test data
scaler = StandardScaler()
test_data_normalized = scaler.fit_transform(test_data_imputed)

# Feature selection for test data
selection = SelectFromModel(loaded_model, prefit=True)
test_data_selected = selection.transform(test_data_normalized)

# Predict the missing column
predicted_column = loaded_model.predict(test_data_selected)

# Add the predicted column to the test data
test_data['Original_NU_NOTA_REDACAO'] = predicted_column

# Save the updated DataFrame
test_data.to_csv('teste_alunos_com_coluna.csv', index=False)
