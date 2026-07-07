from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

# ==========================
# Crear aplicación FastAPI
# ==========================
app = FastAPI(title="Clasificador Inteligente de Residuos")

# Permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Después podremos limitar esto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Cargar modelo UNA sola vez
# ==========================
print("Cargando modelo...")

model = YOLO("best.pt")

print("Modelo cargado correctamente.")

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

    # Ejecutar YOLO
    results = model.predict(imagen)

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