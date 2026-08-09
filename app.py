import os
import subprocess
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(title="Dublikata API NeuralCore v1.5 Ultra")

SECRET_API_KEY = os.environ.get("DUBLIKATA_API_KEY", "dublikata_AI_NeuralCore_ultra:82783893ajdhihgwiudhlkdshw")

BINARY_PATH = os.environ.get("TEXTGEN_BINARY", "./tiny_textgen")
WEIGHTS_PATH = os.environ.get("TEXTGEN_WEIGHTS", "weights.bin")
SOURCE_PATH = "tiny_textgen_int5.cpp"

WEIGHTS_DOWNLOAD_URL = os.environ.get(
    "WEIGHTS_URL", 
    "https://docs.google.com/uc?export=download&id=1Ph5gwo91MD9GC0AReys1TGDUOtobxiWb"
)

def download_weights_and_compile():
    # 1. Скачиваем веса с автоматическим обходом предупреждения Google Диска
    if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_DOWNLOAD_URL:
        print("Скачиваем веса с Google Диска...")
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                response = client.get(WEIGHTS_DOWNLOAD_URL)
                
                # Если Google Диск просит подтверждение для больших файлов
                if "download_warning" in response.text or "confirm=" in response.text:
                    confirm_token = None
                    for key, value in client.cookies.items():
                        if key.startswith("download_warning"):
                            confirm_token = value
                            break
                    
                    if confirm_token:
                        url_with_confirm = f"{WEIGHTS_DOWNLOAD_URL}&confirm={confirm_token}"
                        response = client.get(url_with_confirm)

                response.raise_for_status()
                with open(WEIGHTS_PATH, "wb") as f:
                    f.write(response.content)
            print("Веса успешно скачаны и проверены!")
        except Exception as e:
            print(f"Ошибка скачивания весов: {e}")

    # 2. Проверяем наличие исходников и компилируем
    if not os.path.exists(BINARY_PATH):
        print(f"Поиск файлов: {SOURCE_PATH} -> {os.path.exists(SOURCE_PATH)}, alfavit.cpp -> {os.path.exists('alfavit.cpp')}")
        
        if not os.path.exists(SOURCE_PATH) or not os.path.exists("alfavit.cpp"):
            raise RuntimeError("Файлы исходного кода (.cpp) не найдены в корне проекта на Render!")

        print("Бинарник не найден, запускаем компиляцию...")
        compile_result = subprocess.run(
            ["g++", "-O3", SOURCE_PATH, "alfavit.cpp", "-o", BINARY_PATH],
            capture_output=True,
            text=True
        )
        
        print(f"Return code: {compile_result.returncode}")
        print(f"STDOUT компиляции: {repr(compile_result.stdout)}")
        print(f"STDERR компиляции: {repr(compile_result.stderr)}")
        
        if compile_result.returncode != 0:
            raise RuntimeError(f"КРИТИЧЕСКАЯ ОШИБКА КОМПИЛЯЦИИ:\n{compile_result.stderr}")
        else:
            print("Бинарник успешно скомпилирован на Render!")

# Инициализация при старте сервера
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
        raise HTTPException(status_code=500, detail="Бинарник отсутствует даже после попытки компиляции.")
 
    if not os.path.exists(WEIGHTS_PATH):
        raise HTTPException(status_code=500, detail=f"Файл весов не найден: {WEIGHTS_PATH}.")
 
    try:
        result = subprocess.run(
            [BINARY_PATH],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            error_details = result.stderr.strip() or result.stdout.strip() or "Unknown C++ error"
            raise HTTPException(status_code=500, detail=f"C++ Fail [Code {result.returncode}]: {error_details}")
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="CRASH: Превышено время выполнения бинарника (Timeout)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CRASH Exception: {str(e)}")
 
    return {
        "status": "success",
        "gateway": "Render API Gateway active",
        "echo_prompt": request.prompt,
        "generated": result.stdout.strip(),
    }
