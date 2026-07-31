import os
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Dublikata AI NeuralCore v1.6 (1.1B Int5)")

SECRET_API_KEY = "dublikata_ultra_secret_key_2026"
WEIGHTS_FILE = "dublibrowse_nn_weights_1.1b_int5.bin"
GDRIVE_URL = "https://drive.google.com/file/d/1Ph5gwo91MD9GC0AReys1TGDUOtobxiWb/view?usp=drivesdk"

llm_model = None

def download_file_from_google_drive(url, destination):
    if os.path.exists(destination):
        print(f"[OK] Файл весов уже на месте: {destination}")
        return
    print("[*] Загрузка весов с Google Диска...")
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(download_url, stream=True)
    token = next((val for key, val in response.cookies.items() if key.startswith('download_warning')), None)
    if token:
        response = session.get(download_url, params={'export': 'download', 'confirm': token, 'id': file_id}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"[+] Веса загружены! Размер: {os.path.getsize(destination) / (1024*1024):.2f} MB")

@app.on_event("startup")
def startup_event():
    global llm_model
    try:
        # 1. Скачиваем веса
        download_file_from_google_drive(GDRIVE_URL, WEIGHTS_FILE)
        
        # 2. Инициализируем движок Llama (выделяем под контекст 512 токенов, чтобы экономить RAM)
        print("[*] Инициализация нейросети в памяти (Llama-cpp)...")
        llm_model = Llama(
            model_path=WEIGHTS_FILE,
            n_ctx=512,
            n_threads=2, # Ограничиваем потоки для Render
            verbose=False
        )
        print("[+] Нейросеть успешно запущена и готова к инференсу!")
    except Exception as e:
        print(f"[!] Ошибка инициализации модели: {e}")

@app.get("/")
def root_index():
    status = "online (Model Loaded)" if llm_model else "online (Waiting for weights/RAM)"
    return {"status": status, "core": "Dublikata NeuralCore v1.6 (1.1B Int5)"}

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.post("/generate")
def generate_text(request: PromptRequest, x_api_key: str = Header(...)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Неверный API-ключ Dublikata Studio!")
    
    if not llm_model:
        raise HTTPException(status_code=503, detail="Модель еще загружается в память или произошла ошибка инициализации.")
    
    try:
        user_prompt = request.prompt
        
        # Запускаем честную генерацию через веса
        output = llm_model(
            user_prompt,
            max_tokens=request.max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["<|endoftext|>", "User:", "\n\n\n"]
        )
        
        generated_text = output["choices"][0]["text"].strip()
        thought = f"Dublikata Core [1.1B Int5]: Inferred via Llama engine, tokens generated: {output['usage']['completion_tokens']}"
        
        return {
            "model": "Dublikata NeuralCore v1.6 (1.1B Int5)",
            "thought": thought,
            "response": generated_text if generated_text else "..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
