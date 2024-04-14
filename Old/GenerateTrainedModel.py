import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# Read training and testing data
train_data = pd.read_csv('treinamento_alunos.csv')
test_data = pd.read_csv('teste_alunos.csv')

# Concatenate training and testing data
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# One-hot encode categorical variables
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

# Separate combined data back into training and testing data
train_data_encoded = combined_data_encoded.iloc[:len(train_data)]
test_data_encoded = combined_data_encoded.iloc[len(train_data):].reset_index(drop=True)

# Separate features and target in training data
X_train = train_data_encoded.drop(columns=['Original_NU_NOTA_REDACAO'])
y_train = train_data_encoded['Original_NU_NOTA_REDACAO']

# Impute missing values in training data
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Impute missing values in testing data
X_test_imputed = imputer.transform(test_data_encoded.drop(columns=['Original_NU_NOTA_REDACAO']))

# Make predictions
predictions = model.predict(X_test_imputed)

# Append predictions to the testing DataFrame
test_data['Original_NU_NOTA_REDACAO'] = predictions

# Save the DataFrame with the appended column to a new CSV file
test_data.to_csv('teste_alunos_with_predictions.csv', index=False)

# Make predictions on the training data
y_train_pred = model.predict(X_train_imputed)

# Calculate mean squared error on the training data
mse_train = mean_squared_error(y_train, y_train_pred)

print("Mean Squared Error on the training data:", mse_train)

# Print the predicted values for the testing data
print("Predicted values for the testing data:")
print(predictions)
