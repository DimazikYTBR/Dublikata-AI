from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(title="Dublikata API NeuralCore v1.4 Ultra")

SECRET_API_KEY = "dublikata_AI_NeuralCore_ultra:82783893ajdhihgwiudhlkdshw"

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
    
    # Render теперь просто подтверждает запрос, а всю махинацию с весами делает твой локальный скрипт
    return {
        "status": "success",
        "gateway": "Render API Gateway active",
        "echo_prompt": request.prompt
    }
