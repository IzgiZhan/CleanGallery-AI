from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2
import numpy as np
import io

app = FastAPI(title="CleanGallery AI")

# Подключаем папку для статических файлов (index.html)
app.mount("/static", StaticFiles(directory="."), name="static")

# Главная страница отдаёт HTML
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Проверка размытости
def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

# Проверка темноты
def is_dark(image):
    return np.mean(image) < 40

# Эндпоинт для анализа изображения
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return {
        "blurry": is_blurry(img),
        "dark": is_dark(img)
    }
