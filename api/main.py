import os
import numpy as np
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# Configurações do Hugging Face
REPO_ID = "tiagoparibeiro/melhor_modelo_lstm" # Substitua pelo seu
MODEL_FILENAME = "melhor_modelo_lstm.keras"
SCALER_FILENAME = "melhor_scaler_minmax.joblib"

# Caminhos locais onde o Render/Docker vai salvar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "model")


# Função para garantir que os arquivos existem
def download_assets():
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR)

    # Baixa o modelo
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, local_dir=LOCAL_MODEL_DIR)
    # Baixa o scaler
    scaler_path = hf_hub_download(repo_id=REPO_ID, filename=SCALER_FILENAME, local_dir=LOCAL_MODEL_DIR)

    return model_path, scaler_path


# Exemplo de como usar (para visualização)
print(f"MODEL_PATH: {LOCAL_MODEL_DIR}")

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
async def startup_event():
    global model, scaler
    try:
        model_path, scaler_path = download_assets() # Primeiro garante o download
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        print("Ativos carregados com sucesso do Hugging Face!")
    except Exception as e:
        print(f"Erro no startup: {e}")

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