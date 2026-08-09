import os
import subprocess
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(title="Dublikata API NeuralCore v1.4 Ultra")

SECRET_API_KEY = os.environ.get("DUBLIKATA_API_KEY", "dublikata_AI_NeuralCore_ultra:82783893ajdhihgwiudhlkdshw")

BINARY_PATH = os.environ.get("TEXTGEN_BINARY", "./tiny_textgen")
WEIGHTS_PATH = os.environ.get("TEXTGEN_WEIGHTS", "weights.bin")

# Прямая ссылка для скачивания weights.bin с Google Диска
WEIGHTS_DOWNLOAD_URL = os.environ.get(
    "WEIGHTS_URL", 
    "https://docs.google.com/uc?export=download&id=1Ph5gwo91MD9GC0AReys1TGDUOtobxiWb"
)

def download_weights_if_needed():
    if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_DOWNLOAD_URL:
        print("Файл весов не найден локально, скачиваем с Google Диска...")
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                response = client.get(WEIGHTS_DOWNLOAD_URL)
                response.raise_for_status()
                with open(WEIGHTS_PATH, "wb") as f:
                    f.write(response.content)
            print("Веса успешно скачаны и сохранены!")
        except Exception as e:
            print(f"Ошибка при скачивании весов: {e}")

# Проверяем и скачиваем веса при запуске приложения на Render
download_weights_if_needed()


@app.get("/")
def root_index():
    return {"status": "online", "core": "Dublikata API Gateway (Proxy Mode)"}


class ProxyPromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 150


@app.post("/api/v1/proxy-generate")
def proxy_generate(request: ProxyPromptRequest, x_api_key: str = Header(...)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Неверный API-ключ Dublikata Studio!")
 
    if not os.path.exists(BINARY_PATH):
        raise HTTPException(status_code=500, detail=f"Бинарник не найден: {BINARY_PATH}.")
 
    if not os.path.exists(WEIGHTS_PATH):
        raise HTTPException(status_code=500, detail=f"Файл весов не найден: {WEIGHTS_PATH}.")
 
    try:
        # Добавляем права на исполнение бинарника на всякий случай
        os.chmod(BINARY_PATH, 0o755)
        
        result = subprocess.run(
            [BINARY_PATH],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(os.path.abspath(WEIGHTS_PATH)) or ".",
        )
    except Exception as e:
        # Возвращаем текст ошибки прямо в бот, чтобы увидеть её без лазания по логам
        raise HTTPException(status_code=500, detail=f"Exception: {str(e)}")
 
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Subprocess error: {result.stderr.strip() or result.stdout.strip()}")
 
    return {
        "status": "success",
        "gateway": "Render API Gateway active",
        "echo_prompt": request.prompt,
        "generated": result.stdout.strip(),
    }
