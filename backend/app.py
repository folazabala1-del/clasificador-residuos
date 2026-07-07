from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort
import numpy as np
import io


# ==========================
# Crear aplicación FastAPI
# ==========================
app = FastAPI(title="Clasificador Inteligente de Residuos")


# ==========================
# Configuración CORS
# ==========================
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
# Cargar modelo ONNX
# ==========================
print("Cargando modelo ONNX...")

session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Modelo ONNX cargado correctamente.")


# Clases del modelo
classes = {
    0: "battery",
    1: "biological",
    2: "cardboard",
    3: "clothes",
    4: "glass",
    5: "metal",
    6: "paper",
    7: "plastic",
    8: "sanitary waste and toothbrushes",
    9: "shoes"
}


# ==========================
# Ruta principal
# ==========================
@app.get("/")
def inicio():
    return {
        "mensaje": "API de Clasificación de Residuos funcionando correctamente."
    }


# ==========================
# Health check
# ==========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": "YOLO11n ONNX",
        "estado": "activo"
    }


# ==========================
# Preprocesamiento imagen
# ==========================
def preprocess(image):

    image = image.resize((320,320))

    image = np.array(image)

    image = image[:, :, ::-1]  # RGB a BGR

    image = image.transpose(2,0,1)

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# ==========================
# Predicción
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contenido = await file.read()

    imagen = Image.open(
        io.BytesIO(contenido)
    ).convert("RGB")


    # Preparar tensor
    tensor = preprocess(imagen)


    # Ejecutar ONNX
    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )


    detecciones = []


    # Salida YOLO11:
    # (1,14,2100)
    predictions = outputs[0][0]


    predictions = predictions.transpose()


    for pred in predictions:

        x, y, w, h = pred[:4]

        scores = pred[4:]

        class_id = np.argmax(scores)

        confidence = scores[class_id]


        if confidence > 0.5:

            detecciones.append({

                "class": classes.get(
                    int(class_id),
                    "unknown"
                ),

                "confidence": round(
                    float(confidence),
                    3
                ),

                "bbox":[
                    round(float(x),1),
                    round(float(y),1),
                    round(float(w),1),
                    round(float(h),1)
                ]
            })


    return {
        "detections": detecciones
    }