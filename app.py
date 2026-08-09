import os
import subprocess
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(title="Dublikata API NeuralCore v1.4 Ultra")

SECRET_API_KEY = os.environ.get("DUBLIKATA_API_KEY", "dublikata_AI_NeuralCore_ultra:82783893ajdhihgwiudhlkdshw")

BINARY_PATH = os.environ.get("TEXTGEN_BINARY", "./tiny_textgen")
WEIGHTS_PATH = os.environ.get("TEXTGEN_WEIGHTS", "weights.bin")
SOURCE_PATH = "tiny_textgen_int5.cpp"

WEIGHTS_DOWNLOAD_URL = os.environ.get(
    "WEIGHTS_URL", 
    "https://docs.google.com/uc?export=download&id=1Ph5gwo91MD9GC0AReys1TGDUOtobxiWb"
)

def download_weights_and_compile():
    # 1. Скачиваем веса, если их нет
    if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_DOWNLOAD_URL:
        print("Скачиваем веса с Google Диска...")
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                response = client.get(WEIGHTS_DOWNLOAD_URL)
                response.raise_for_status()
                with open(WEIGHTS_PATH, "wb") as f:
                    f.write(response.content)
            print("Веса успешно скачаны!")
        except Exception as e:
            print(f"Ошибка скачивания весов: {e}")

    # 2. Если бинарника нет, но есть исходник на C++ — компилируем его прямо на сервере!
    if not os.path.exists(BINARY_PATH) and os.path.exists(SOURCE_PATH):
        print("Бинарник не найден, компилируем из исходника...")
        compile_result = subprocess.run(
            ["g++", "-O3", SOURCE_PATH, "-o", BINARY_PATH],
            capture_output=True,
            text=True
        )
        if compile_result.returncode != 0:
            print(f"Ошибка компиляции: {compile_result.stderr}")
        else:
            print("Бинарник успешно скомпилирован на Render!")

# Инициализация при старте
download_weights_and_compile()


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
        raise HTTPException(status_code=500, detail=f"Бинарник отсутствует даже после попытки компиляции.")
 
    if not os.path.exists(WEIGHTS_PATH):
        raise HTTPException(status_code=500, detail=f"Файл весов не найден: {WEIGHTS_PATH}.")
 
    except Exception as e:
        # Возвращаем текст ошибки СТРОГО в поле detail, чтобы бот её показал
        raise HTTPException(status_code=500, detail=f"CRASH: {str(e)}")


    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Subprocess error: {result.stderr.strip() or result.stdout.strip()}")
 
    return {
        "status": "success",
        "gateway": "Render API Gateway active",
        "echo_prompt": request.prompt,
        "generated": result.stdout.strip(),
    }
