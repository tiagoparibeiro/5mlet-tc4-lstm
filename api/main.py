import os
import numpy as np
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
from tensorflow.keras.models import load_model

## 1. Obtém o caminho absoluto do arquivo atual
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Navega para o diretório pai (cd ..)
#    Se o script está em 'api/', o diretório pai é a raiz do projeto
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 3. Junta o diretório raiz com o caminho relativo desejado
MODEL_PATH = os.path.join(PROJECT_ROOT, "notebooks", "melhor_modelo_lstm.keras")
SCALER_PATH = os.path.join(PROJECT_ROOT, "notebooks", "melhor_scaler_minmax.joblib")

# Exemplo de como usar (para visualização)
print(f"MODEL_PATH: {MODEL_PATH}")

WINDOW_SIZE = 30
WINDOW_SIZE = 30
TARGET_COL_INDEX = 0
N_FEATURES = 5

app = FastAPI(
    title="Time Series Prediction API",
    description="API para prever valores de séries temporais usando modelo LSTM.",
    version = "1.0.0"
)

model = None
scaler = None

#class FeaturesVector(BaseModel):
#    root: list[float]

class PredictionInput(BaseModel):
    data: list[list[float]]

    @property
    def window_size_ok(self):
        return len(self.data) == WINDOW_SIZE

@app.on_event("startup")
def load_assets():
    """Carrega o modelo e o scaler na inicialização da API."""
    global model, scaler
    try:
        model = load_model(MODEL_PATH)
        print(f"Modelo carregado com sucesso de: {MODEL_PATH}")

        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler carregado com sucesso de : {SCALER_PATH}")

    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        raise RuntimeError(f"Erro ao carregar modelo ou scaler: {e}")

@app.get("/health", tags=["Monitoring"])
def get_health():
    """Verifica o status da API e se o modelo foi carregado"""
    status = "ok" if model is not None and scaler is not None else "degraded"
    return {"status": status, "model_loaded": model is not None, "scaler_loaded": scaler is not None}

@app.post("/predict", tags=["Prediction"])
def predict_next_step(input_data: PredictionInput):
    """Recebe os dados da janela de tempo e retorna a previsão desnormalizada"""
    if not input_data.window_size_ok:
        raise HTTPException(
            status_code=400,
            detail = f"Tamanho da janela de entrada inválido. Esperado: {WINDOW_SIZE}, RECEBIDO {len(input_data.data)}"
        )

    try:
        input_2d = np.array(input_data.data)

        if input_2d.shape[1] != N_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Numero de features inválido. Esperado {N_FEATURES}, Recebido {input_2d.shape[1]}"
            )

        input_scaled = scaler.transform(input_2d)

        X_inference = np.expand_dims(input_scaled, axis=0)

        y_pred_scaled = model.predict(X_inference)

        dummy_pred = np.zeros((1, N_FEATURES))
        dummy_pred[0, TARGET_COL_INDEX] = y_pred_scaled[0, 0]

        y_pred_original = scaler.inverse_transform(dummy_pred)[0, TARGET_COL_INDEX]

        return {
            "prediction": float(y_pred_original),
            "unit": "Valor Original da Coluna Alvo"
        }
    except Exception as e:
        print(f"Erro durante a inferência: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {e}")

    # 5. Execução do Servidor (Para testes locais)
if __name__ == "__main__":
    # Este é o comando para rodar localmente. Use 'uvicorn api.main:app --reload'
    # no terminal para uso em desenvolvimento.
    uvicorn.run(app, host="0.0.0.0", port=8000)