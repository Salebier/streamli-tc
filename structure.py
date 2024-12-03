import torch
import torch.nn as nn

# Definição de um mecanismo de atenção como classe para uso em redes neurais
class Attention(nn.Module):
    def __init__(self, hidden_size):
        """
        Inicializa o mecanismo de atenção.
        Parâmetros:
        - hidden_size: tamanho das camadas ocultas do LSTM, usado para definir as dimensões da atenção.
        """

        super(Attention, self).__init__()

        # Camada linear para calcular os escores de atenção
        # Multiplica (hidden_size * 2) dimensões das saídas do LSTM por 1 (resultado escalar por passo temporal)
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        """
        Executa o mecanismo de atenção sobre as saídas do LSTM.
        Parâmetro:
        - lstm_output: tensor com saídas do LSTM (batch_size, seq_len, hidden_size * 2).
        Retorna:
        - context_vector: vetor de contexto resultante da soma ponderada (batch_size, hidden_size * 2).
        - attention_weights: pesos de atenção normalizados (batch_size, seq_len).
        """

        # Calcula os escores de atenção para cada passo temporal
        # Saída: (batch_size, seq_len, 1)
        scores = self.attention(lstm_output)

        # Remove a dimensão extra (última dimensão) para facilitar operações posteriores
        # Saída: (batch_size, seq_len)
        scores = scores.squeeze(-1)

        # Aplica a função softmax para normalizar os escores ao longo do eixo da sequência (dim=1)
        # Gera os pesos de atenção: valores no intervalo [0, 1], cuja soma é 1 para cada sequência
        attention_weights = torch.softmax(scores, dim=1)

        # Calcula o vetor de contexto como uma soma ponderada das saídas do LSTM
        # Multiplica cada saída do LSTM pelos pesos de atenção correspondentes
        # `attention_weights.unsqueeze(-1)` expande os pesos para alinhar com as dimensões do LSTM
        # Saída: (batch_size, hidden_size * 2)
        context_vector = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)

        # Retorna o vetor de contexto e os pesos de atenção
        return context_vector, attention_weights
    
# Modelo LSTM desenvolvido para prever valores de fechamento entre os valores máximos e mínimos de cada dia
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Inicializa o modelo LSTM com atenção.
        Parâmetros:
        - input_size: número de características (features) de entrada.
        - hidden_size: tamanho das camadas ocultas do LSTM.
        - num_layers: número de camadas LSTM empilhadas.
        - output_size: tamanho da saída final (geralmente 1 para regressão de valores únicos).
        """
        super(LSTMModel, self).__init__()

        # Definição do LSTM bidirecional
        # batch_first=True: garante que a entrada tenha a forma (batch_size, seq_len, input_size)
        # bidirectional=True: permite que o LSTM processe sequências em ambas as direções (frente e trás)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Mecanismo de atenção, definido como uma camada separada
        # hidden_size é usado para calcular escores de atenção
        self.attention = Attention(hidden_size)

        # Camada totalmente conectada (FC) para processar o vetor de contexto da atenção
        # Inclui duas camadas ReLU e Dropout para prevenir overfitting
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Entrada: vetor bidirecional do LSTM
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout para regularização
            nn.Linear(hidden_size, hidden_size // 2),  # Redução das dimensões intermediárias
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)  # Saída: valor final de previsão
        )

    def forward(self, x, high, low):
        """
        Propagação direta do modelo.
        Parâmetros:
        - x: tensor de entrada com forma (batch_size, seq_len, input_size).
        - high: tensor com valores máximos para normalização (batch_size, 1).
        - low: tensor com valores mínimos para normalização (batch_size, 1).
        Retorna:
        - out: tensor com os valores previstos (batch_size, output_size).
        - attention_weights: pesos de atenção calculados para cada passo temporal.
        """

        # Inicializando os estados ocultos e de célula com zeros
        # `num_layers * 2` por ser um LSTM bidirecional
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)

        # Propagação dos dados através do LSTM
        # `out` contém as saídas do LSTM para todos os passos temporais
        out, _ = self.lstm(x, (h0, c0))

        # Aplicando o mecanismo de atenção nas saídas do LSTM
        # `context_vector` é a soma ponderada das saídas do LSTM
        # `attention_weights` são os pesos de atenção calculados
        context_vector, attention_weights = self.attention(out)

        # Passa o vetor de contexto pelas camadas totalmente conectadas (FC)
        out = self.fc(context_vector)

        # Normaliza a saída para o intervalo [0, 1] usando a função sigmoid
        out = torch.sigmoid(out)

        # Ajusta a saída para o intervalo entre `low` e `high` (valores mínimos e máximos do dia)
        # Fórmula: low + (high - low) * out (interpolação linear)
        out = low + (high - low) * out

        # Retorna a previsão final e os pesos de atenção
        return out, attention_weights