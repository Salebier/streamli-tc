import pandas as pd
import streamlit as st
import pickle
import numpy as np

import torch
from structure import Attention, LSTMModel
from sklearn.preprocessing import MinMaxScaler


# Título
st.write("""
Prevendo Possível Preço da Ação da NVIDIA
""")

# Título sidebar
st.sidebar.title('Preencha os Dados')


# Função para obter os dados do usuário
def get_user_data():
    open = st.sidebar.number_input('Open', min_value=0.0, max_value=10000.0, value=0.0, step=0.1)
    high = st.sidebar.number_input('High', min_value=0.0, max_value=10000.0, value=0.0, step=0.1)
    low = st.sidebar.number_input('Low', min_value=0.0, max_value=10000.0, value=0.0, step=0.1)


    user_data = {
        'Open': open,
        'High': high,
        'Low': low
    }

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_data()

# Gráfico
# graf = st.bar_chart(user_input_variables)

st.subheader('Dados fornecidos pelo usuário:')
st.write(user_input_variables)

# Carregar o modelo salvo com pickle
with open('LSTM_treinado_modelo.pkl', 'rb') as file:
    model = pickle.load(file)

with open('LSTM_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Colocar o modelo em modo de avaliação
model.eval()

# Dados de entrada como lista de valores (uma única linha)
raw_input = [[user_input_variables['Open'].values[0], user_input_variables['High'].values[0], user_input_variables['Low'].values[0], 0]]  # Open, High, Low, Close

# Transformar os dados de entrada para o formato esperado pelo scaler
scaled_input = scaler.transform(raw_input)  # Escala todas as features

# Repetir o valor 20 vezes para criar a sequência
sequence_input = torch.tensor(scaled_input).repeat(20, 1).unsqueeze(0).float()  # (batch_size=1, seq_len=20, input_size=4)

# Extrair valores de High e Low da sequência para ajuste na saída
scaled_high = sequence_input[:, :, 1:2]  # Coluna de High
scaled_low = sequence_input[:, :, 2:3]   # Coluna de Low


# Fazer a previsão
with torch.no_grad():  # Desativa o cálculo de gradientes
    prediction, attention_weights = model(sequence_input, scaled_high, scaled_low)


# Converter o tensor de predições para numpy e ajustar a forma
predictions_numpy = prediction.squeeze(-1).detach().numpy().reshape(-1, 1)  # (20, 1)

# Preencher as outras 3 colunas com valores das features "Open", "High", "Low"
predictions_extended = np.repeat(predictions_numpy, 4, axis=1)  # (20, 4)

# Reverter a normalização para as 4 features
predicted_prices = scaler.inverse_transform(predictions_extended)

# Converter para lista
predictions_list = predicted_prices[:, -1].tolist()  # Pega apenas "Close"


st.subheader('📊Previsão: ')
st.write(predictions_list[0])


prediction_value = predictions_list[0]
user_input_variables['Predicao'] = prediction_value

# Exibir a tabela atualizada com o valor de previsão
st.subheader('Dados fornecidos pelo usuário mais previsão:')
st.write(user_input_variables)