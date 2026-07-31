import os
import json
from fastapi import FastAPI, HTTPException, Security, Header
from pydantic import BaseModel

app = FastAPI(title="Dublikata AI NeuralCore v1.4 Ultra")

# Секретный ключ для защиты API
SECRET_API_KEY = "dublikata_ultra_secret_key_2026"
WEIGHTS_FILE = "dublibrowse_nn_weights_1.0b_int8.bin"

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Неверный API-ключ Dublikata Studio!")
    return x_api_key

@app.on_event("startup")
def load_weights_metadata():
    # Проверяем локально или на подключенном Диске
    target_path = WEIGHTS_FILE
    if not os.path.exists(target_path):
        # Пробуем найти на Google Диске, если запущены в Colab
        drive_path = os.path.join('/content/drive/MyDrive/DublikataAI', WEIGHTS_FILE)
        if os.path.exists(drive_path):
            target_path = drive_path

    if os.path.exists(target_path):
        with open(target_path, "rb") as f:
            header_data = f.read(512).decode("utf-8", errors="ignore").strip()
            print(f"NeuralCore успешно активирован! Метаданные: {header_data[:120]}...")
    else:
        print("Внимание: Файл весов не обнаружен!")

@app.get("/")
def home():
    return {"status": "Dublikata AI NeuralCore v1.4 Ultra is online and ready 24/7!"}

@app.post("/generate")
def generate_text(request: PromptRequest, api_key: str = Security(verify_api_key)):
    try:
        # Логика генерации ответа на основе промпта
        return {
            "model": "Dublikata AI NeuralCore v1.4 Ultra",
            "received_prompt": request.prompt,
            "response": f"[NeuralCore v1.4 Ultra]: Успешно сгенерировано ответ на ваш запрос в DubliBrowse."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
