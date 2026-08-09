import os
import subprocess
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
 
app = FastAPI(title="Dublikata API NeuralCore v1.4 Ultra")
 
# Лучше вынести в переменную окружения Render (Environment -> Add env var),
# а не хранить прямо в коде — так ключ не утечёт вместе с репозиторием.
SECRET_API_KEY = os.environ.get("DUBLIKATA_API_KEY", "dublikata_AI_NeuralCore_ultra:82783893ajdhihgwiudhlkdshw")
 
# Путь к скомпилированному бинарнику. Собирается на этапе Build Command (см. ниже).
BINARY_PATH = os.environ.get("TEXTGEN_BINARY", "./tiny_textgen")
WEIGHTS_PATH = os.environ.get("TEXTGEN_WEIGHTS", "weights.bin")
 
 
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
        raise HTTPException(status_code=500, detail=f"Бинарник не найден: {BINARY_PATH}. Проверь Build Command на Render.")
 
    if not os.path.exists(WEIGHTS_PATH):
        raise HTTPException(status_code=500, detail=f"Файл весов не найден: {WEIGHTS_PATH}.")
 
    try:
        # Бинарник сейчас не принимает prompt/max_tokens как аргументы —
        # он генерирует фиксированные 100 символов от seed_token='H'.
        # Это просто запуск процесса и чтение его stdout.
        result = subprocess.run(
            [BINARY_PATH],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(os.path.abspath(WEIGHTS_PATH)) or ".",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Генерация не уложилась в таймаут")
 
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {result.stderr.strip()}")
 
    return {
        "status": "success",
        "gateway": "Render API Gateway active",
        "echo_prompt": request.prompt,
        "generated": result.stdout.strip(),
    }
