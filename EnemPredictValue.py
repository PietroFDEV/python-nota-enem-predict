## Pietro Goudel Favoreto

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Ler arquivos
train_data = pd.read_csv('treinamento_alunos.csv')
test_data = pd.read_csv('teste_alunos.csv')

# Concatenamento dos 2 arquivos
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Mapeamento das colunas conhecidas do excel
combined_data_encoded = pd.get_dummies(combined_data, columns=[
    'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'TP_COR_RACA',
    'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA',
    'TP_ENSINO', 'IN_TREINEIRO', 'CO_MUNICIPIO_ESC', 'CO_UF_ESC', 'SG_UF_ESC',
    'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC', 'TP_SIT_FUNC_ESC',
    'CO_MUNICIPIO_PROVA', 'CO_UF_PROVA', 'SG_UF_PROVA', 'TP_PRESENCA_CN',
    'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'CO_PROVA_CN',
    'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT', 'TP_LINGUA',
    'TP_STATUS_REDACAO', 'Q001', 'Q002', 'Q003', 'Q004', 'Q006', 'Q007',
    'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016',
    'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025'
])

# Criar variáveis para cada arquivo CSV
train_data_encoded = combined_data_encoded.iloc[:len(train_data)]
test_data_encoded = combined_data_encoded.iloc[len(train_data):].reset_index(drop=True)

# Separar colunas, e objetivo do treinamento do código
X_train = train_data_encoded.drop(columns=['Original_NU_NOTA_REDACAO'])
y_train = train_data_encoded['Original_NU_NOTA_REDACAO']

# Colocar valores, caso tenha algo faltando dado
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Treinamento do modelo de Lasso
model = Lasso(alpha=0.1)  # Este Alpha value funcionou ok
model.fit(X_train_imputed, y_train)

# Colocar dados, para campos sem valores, pois estava dando erro na coluna inexistente
X_test_imputed = imputer.transform(test_data_encoded.drop(columns=['Original_NU_NOTA_REDACAO']))

# Realizar as previsões
predictions = model.predict(X_test_imputed)

# Adicionar a nova coluna com as previsões
test_data['NU_NOTA_REDACAO'] = predictions

# Salvar um novo CSV com as mudanças
test_data.to_csv('teste_alunos_with_predictions.csv', index=False)

# Realizar as previsões no modelo de teste
y_train_pred = model.predict(X_train_imputed)

# Para assim verificar a raiz do erro
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

print("Root Mean Squared Error on the training data:", rmse_train)