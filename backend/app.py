from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort
import numpy as np
import io


# ==========================
# Crear aplicación FastAPI
# ==========================
app = FastAPI(
    title="Clasificador Inteligente de Residuos"
)


# ==========================
# CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        # agrega aquí la URL de tu frontend desplegado cuando la tengas
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# Cargar modelo ONNX
# ==========================
print("Cargando modelo ONNX...")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "best.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
TAMANO_ENTRADA = 320  # debe coincidir con el tamaño de exportación del ONNX

# "Calentamos" el modelo para que la primera petición real de un usuario
# no pague el costo de inicialización interna de ONNX Runtime.
print("Calentando el modelo...")
dummy_input = np.zeros((1, 3, TAMANO_ENTRADA, TAMANO_ENTRADA), dtype=np.float32)
session.run(None, {input_name: dummy_input})
print("Modelo ONNX cargado y listo.")


# ==========================
# Clases del modelo
# ==========================
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
# NMS
# ==========================
def nms(boxes, scores, threshold=0.45):
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou < threshold)[0]
        order = order[inds + 1]

    return keep


# ==========================
# Preprocesamiento (CORREGIDO: letterbox en vez de resize directo)
# ==========================
def preprocess(image, tamano=TAMANO_ENTRADA, color=(114, 114, 114)):
    """
    Redimensiona la imagen MANTENIENDO su proporción original (a
    diferencia de un resize directo, que la deforma), y rellena el
    espacio sobrante con un color gris neutro. Esta es la misma técnica
    (letterbox) que usa Ultralytics internamente durante el entrenamiento
    y la validación.

    Devuelve el tensor listo para el modelo, más la escala y el padding
    aplicados, necesarios para convertir las cajas de vuelta a las
    coordenadas de la imagen original.
    """
    ancho_original, alto_original = image.size

    escala = min(tamano / ancho_original, tamano / alto_original)
    nuevo_ancho = round(ancho_original * escala)
    nuevo_alto = round(alto_original * escala)

    imagen_redimensionada = image.resize((nuevo_ancho, nuevo_alto))

    lienzo = Image.new("RGB", (tamano, tamano), color)
    pad_x = (tamano - nuevo_ancho) // 2
    pad_y = (tamano - nuevo_alto) // 2
    lienzo.paste(imagen_redimensionada, (pad_x, pad_y))

    tensor = np.array(lienzo).astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, escala, pad_x, pad_y


# ==========================
# Inicio
# ==========================
@app.get("/")
def inicio():
    return {
        "mensaje": "API de Clasificación de Residuos funcionando correctamente."
    }


# ==========================
# Health
# ==========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": "YOLO11n ONNX",
        "estado": "activo"
    }


# ==========================
# Predicción
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contenido = await file.read()
    imagen = Image.open(io.BytesIO(contenido)).convert("RGB")

    tensor, escala, pad_x, pad_y = preprocess(imagen)

    outputs = session.run(None, {input_name: tensor})
    predictions = outputs[0][0]
    predictions = predictions.transpose()

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        cx, cy, w, h = pred[:4]
        class_scores = pred[4:]

        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > 0.5:
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Deshacemos el letterbox: quitamos el padding y revertimos
            # la escala, para volver a las coordenadas de la imagen
            # ORIGINAL que mandó el navegador.
            x1 = (x1 - pad_x) / escala
            y1 = (y1 - pad_y) / escala
            x2 = (x2 - pad_x) / escala
            y2 = (y2 - pad_y) / escala

            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            scores.append(float(confidence))
            class_ids.append(int(class_id))

    detecciones = []

    if len(boxes) > 0:
        keep = nms(boxes, scores)
        for i in keep:
            detecciones.append({
                "class": classes[class_ids[i]],
                "confidence": round(scores[i], 3),
                "bbox": [round(v, 1) for v in boxes[i]],
            })

    return {"detections": detecciones}