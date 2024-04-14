import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Carregar os dados de teste
dados_teste = pd.read_csv('teste_alunos.csv')

# Carregar o modelo treinado
lasso_model = joblib.load('lasso_model.pkl')

# Pré-processamento dos dados de teste
# Remove a coluna 'Original_NU_NOTA_REDACAO' (se existir)
if 'Original_NU_NOTA_REDACAO' in dados_teste.columns:
    dados_teste.drop(columns=['Original_NU_NOTA_REDACAO'], inplace=True)

# Combinação de features
dados_teste['MEDIA_NOTAS'] = (dados_teste['NU_NOTA_CN'] + dados_teste['NU_NOTA_CH'] + dados_teste['NU_NOTA_LC'] + dados_teste['NU_NOTA_MT']) / 4
dados_teste['SOMA_NOTAS'] = dados_teste['NU_NOTA_CN'] + dados_teste['NU_NOTA_CH'] + dados_teste['NU_NOTA_LC'] + dados_teste['NU_NOTA_MT']
dados_teste['DIFERENCA_MT_LC'] = dados_teste['NU_NOTA_MT'] - dados_teste['NU_NOTA_LC']

# Transformações numéricas
dados_teste['LOG_MEDIA_NOTAS'] = np.log1p(dados_teste['MEDIA_NOTAS'])
dados_teste['RAIZ_SOMA_NOTAS'] = np.sqrt(dados_teste['SOMA_NOTAS'])

# Codificar variáveis categóricas
dados_teste = pd.get_dummies(dados_teste)

# Carregar o imputer e o scaler do modelo treinado
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Aplicar as transformações ao conjunto de teste
dados_teste_imputados = imputer.transform(dados_teste)
dados_teste_scaled = scaler.transform(dados_teste_imputados)

# Aplicar o modelo aos dados de teste para fazer previsões
previsoes_teste = lasso_model.predict(dados_teste_scaled)

# Salvar as previsões em um arquivo ou utilizar conforme necessário
