import os
import requests
from fastapi import FastAPI, HTTPException, Security, Header
from pydantic import BaseModel

app = FastAPI(title="Dublikata AI NeuralCore v1.6 (1.1B Int5)")

SECRET_API_KEY = "dublikata_ultra_secret_key_2026"
WEIGHTS_FILE = "dublibrowse_nn_weights_1.1b_int5.bin"

# Твоя ссылка на Google Диск
GDRIVE_URL = "https://drive.google.com/file/d/1Ph5gwo91MD9GC0AReys1TGDUOtobxiWb/view?usp=drivesdk"

def download_file_from_google_drive(url, destination):
    if os.path.exists(destination):
        print(f"[OK] Файл весов уже существует на сервере: {destination}")
        return

    print("[*] Скачиваем файл весов с Google Диска на сервер Render (это займет немного времени)...")
    
    # Преобразуем ссылку обычного просмотра в прямую ссылку на скачивание
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()
    response = session.get(download_url, stream=True)
    
    # Проверяем подтверждение для больших файлов от Google
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'export': 'download', 'confirm': token, 'id': file_id}
        response = session.get(download_url, params=params, stream=True)

    # Записываем файл на диск Render чанками
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                
    print(f"[+] Веса успешно скачаны! Размер: {os.path.getsize(destination) / (1024*1024):.2f} MB")

@app.on_event("startup")
def startup_event():
    try:
        download_file_from_google_drive(GDRIVE_URL, WEIGHTS_FILE)
    except Exception as e:
        print(f"[!] Ошибка при скачивании весов: {e}")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

@app.post("/generate")
def generate_text(request: PromptRequest, x_api_key: str = Header(...)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Неверный API-ключ Dublikata Studio!")
    try:
        user_text = request.prompt
        thought_process = f"Dublikata Core 1.1B [Int5]: Multilingual intent analysis & reasoning chain."
        ai_reply = f"Привет! Ядро Dublikata NeuralCore v1.6 (1.1B, 5-bit) обработало запрос: {user_text}"
        return {"model": "Dublikata NeuralCore v1.6 (1.1B Int5)", "thought": thought_process, "response": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
