from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

# ==========================
# Crear aplicación FastAPI
# ==========================
app = FastAPI(title="Clasificador Inteligente de Residuos")

# Permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Cargar modelo UNA sola vez
# ==========================
print("Cargando modelo...")

model = YOLO("best.pt")

print("Modelo cargado correctamente.")

# Preparar modelo para evitar demora en la primera inferencia
model.predict(
    source=np.zeros((320,320,3), dtype=np.uint8),
    imgsz=320,
    device="cpu",
    verbose=False
)

print("Modelo preparado.")

# ==========================
# Ruta principal
# ==========================
@app.get("/")
def inicio():
    return {
        "mensaje": "API de Clasificación de Residuos funcionando correctamente."
    }


# ==========================
# Estado de la API
# ==========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": "YOLO11n",
        "estado": "activo"
    }


# ==========================
# Predicción
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Leer imagen enviada
    contenido = await file.read()

    imagen = Image.open(io.BytesIO(contenido))

    # Reducir tamaño para Render
    imagen.thumbnail((640,640))


    # Ejecutar YOLO
    results = model.predict(
    imagen,
    imgsz=320,
    device="cpu",
    verbose=False
)
    detecciones = []

    for result in results:

        for box in result.boxes:

            clase = result.names[int(box.cls)]

            confianza = float(box.conf)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detecciones.append({
                "class": clase,
                "confidence": round(confianza, 3),
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1)
                ]
            })

    return {
        "detections": detecciones
    }