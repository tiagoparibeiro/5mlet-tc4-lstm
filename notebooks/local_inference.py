import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

SCALER_FILENAME = 'scaler_minmax_melhor.joblib'

try:
    # Carrega o objeto scaler
    scaler = joblib.load(SCALER_FILENAME)
    print(f"✅ Scaler carregado com sucesso de: {SCALER_FILENAME}")
except FileNotFoundError:
    print(f"❌ Erro: O arquivo do scaler '{SCALER_FILENAME}' não foi encontrado.")
    exit()


# --- Configurações que DEVEM ser as mesmas do treinamento ---
MODEL_FILENAME = 'modelo_lstm_series_temporais.keras'
WINDOW_SIZE = 30  # Substitua pelo valor real da sua 'window'
TARGET_COLUMN_INDEX = 0  # Substitua pelo índice real da sua 'target_col_index'
N_FEATURES = 3  # Substitua pelo número real de colunas em seu 'df' original

# -----------------------------------------------------------

## 1. Carregar o Modelo
try:
    print(f"Carregando modelo do arquivo: {MODEL_FILENAME}...")
    model = load_model(MODEL_FILENAME)
    print("✅ Modelo carregado com sucesso.")
except FileNotFoundError:
    print(f"❌ Erro: O arquivo do modelo '{MODEL_FILENAME}' não foi encontrado.")
    exit()


## 2. Função de Preparação de Dados (Simulação)
# Na prática, você carregaria os últimos N_FEATURES da sua fonte de dados.
# Aqui, simulamos carregando os últimos dados do seu DataFrame original (df).

def load_inference_data(df, window):
    """
    Carrega os últimos 'window' pontos de dados do DataFrame
    para usar como entrada para a previsão.
    """
    if len(df) < window:
        raise ValueError(f"DataFrame é muito pequeno. Precisa de pelo menos {window} linhas.")

    # Pega as últimas 'window' linhas
    last_data = df.iloc[-window:]

    print(f"Dados de entrada (últimas {window} observações):\n")
    print(last_data)

    return last_data.values  # Retorna como array numpy


## 3. Realizar a Inferência

def perform_inference(input_data_2d, model, window, target_idx, n_features):
    # 3.1. Normalização (Fit não é necessário, apenas Transform!)
    # Atenção: O MinMaxScaler deve ser *reinstanciado e fitado* com os dados
    # que foram usados para o treinamento original (ou use um scaler salvo).
    # **Como você não salvou o scaler, vamos SIMULAR que ele foi fitado no DF inteiro
    # Na produção, você **DEVE** salvar o scaler junto com o modelo.

    # SIMULAÇÃO DA CRIAÇÃO DO SCALER (Apenas para que o `inverse_transform` funcione)
    # Para uma inferência correta, o SCALER original (fitado no treino) DEVE ser carregado.
    temp_scaler = MinMaxScaler()

    # Se você tivesse o DataFrame `df` original (completo), faria:
    # temp_scaler.fit(df.values)

    # Como não temos, vamos simular os limites de normalização.
    # Esta é a parte MAIS FRÁGIL de um script de inferência.
    # Para ser robusto, o scaler deve ser salvo (ex: via joblib ou pickle).

    # Para este exemplo funcionar, vou supor que o SCALER foi fitado em algo que
    # gerou dados de validação entre 0 e 1, e vamos usá-lo para reverter.
    # Isso só funcionará se você tiver o objeto `scaler` original.
    # Se você **NÃO SALVOU O SCALER**, você **NÃO PODE** normalizar e desnormalizar corretamente.

    # **AVISO:** Este código pressupõe que você tem acesso ao `scaler` que foi
    # `fitado` nos dados de treino. Se não tem, a normalização/desnormalização falhará.

    # --- PREPARANDO O INPUT (SE VOCÊ TIVER O SCALER ORIGINAL) ---

    # Se você *tivesse* o scaler original:
    # input_scaled = scaler.transform(input_data_2d)

    # --- SIMULAÇÃO: Se você tivesse o `df` (dataframe original):
    # Se você está no mesmo ambiente e tem `df` e `scaler` do treino:
    scaler.fit(df.values)  # <-- Essa linha só funciona se `df` estiver carregado
    input_scaled = scaler.transform(input_data_2d)

    # 3.2. Formatar para LSTM (Janela)
    # A entrada deve ter o formato (1, window, n_features)
    X_inference = np.expand_dims(input_scaled, axis=0)

    # 3.3. Previsão
    print("\nRealizando previsão...")
    y_pred_scaled = model.predict(X_inference)

    # 3.4. Inverter a Normalização
    # Criar um array dummy para o inverse_transform
    dummy_pred = np.zeros((len(y_pred_scaled), n_features))

    # Colocar a previsão no índice da coluna alvo
    # y_pred_scaled tem shape (1, 1), então pegamos [0, 0]
    dummy_pred[:, target_idx] = y_pred_scaled[0, 0]

    # Inverter a normalização
    y_pred_original = scaler.inverse_transform(dummy_pred)[:, target_idx][0]

    return y_pred_original


# --- EXECUTANDO A INFERÊNCIA (Você deve garantir que 'df' e 'scaler' estejam disponíveis) ---

# Atenção: Você precisa ter o DataFrame 'df' (com pelo menos WINDOW_SIZE linhas) e o objeto 'scaler'
# (treinado no set de treino original) disponíveis neste script para que a
# normalização e desnormalização funcionem corretamente.

try:
    # 1. Carregar novos dados (Simulando usando o final do DataFrame 'df' do treino)
    # **NOTA**: Certifique-se de que 'df' está disponível neste ambiente.
    new_data = load_inference_data(df, WINDOW_SIZE)

    # 2. Executar a Inferência
    prediction = perform_inference(
        input_data_2d=new_data,
        model=model,
        window=WINDOW_SIZE,
        target_idx=TARGET_COLUMN_INDEX,
        n_features=N_FEATURES
    )

    print("\n==============================================")
    print(f"🌟 Próxima Previsão: {prediction:.4f}")
    print("==============================================")

except NameError:
    print("\n--- ERRO CRÍTICO DE AMBIENTE ---")
    print("As variáveis 'df' e/ou 'scaler' não foram definidas no ambiente.")
    print(
        "Você precisa carregar o DataFrame (df) e o Scaler (scaler) que foram usados no treinamento para que a inferência funcione.")
except Exception as e:
    print(f"Ocorreu um erro durante a inferência: {e}")