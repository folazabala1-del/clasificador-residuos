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
    providers=[
        "CPUExecutionProvider"
    ]
)


input_name = session.get_inputs()[0].name


print("Modelo ONNX cargado correctamente.")


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


    areas = (
        x2 - x1
    ) * (
        y2 - y1
    )


    order = scores.argsort()[::-1]

    keep = []


    while len(order) > 0:

        i = order[0]

        keep.append(i)


        xx1 = np.maximum(
            x1[i],
            x1[order[1:]]
        )

        yy1 = np.maximum(
            y1[i],
            y1[order[1:]]
        )


        xx2 = np.minimum(
            x2[i],
            x2[order[1:]]
        )

        yy2 = np.minimum(
            y2[i],
            y2[order[1:]]
        )


        w = np.maximum(
            0,
            xx2 - xx1
        )

        h = np.maximum(
            0,
            yy2 - yy1
        )


        inter = w * h


        iou = inter / (
            areas[i]
            +
            areas[order[1:]]
            -
            inter
            +
            1e-6
        )


        inds = np.where(
            iou < threshold
        )[0]


        order = order[inds + 1]


    return keep



# ==========================
# Preprocesamiento
# ==========================
def preprocess(image):

    image = image.resize(
        (320,320)
    )


    image = np.array(
        image
    )


    image = image.astype(
        np.float32
    ) / 255.0


    image = image.transpose(
        2,
        0,
        1
    )


    image = np.expand_dims(
        image,
        axis=0
    )


    return image



# ==========================
# Inicio
# ==========================
@app.get("/")
def inicio():

    return {
        "mensaje":
        "API de Clasificación de Residuos funcionando correctamente."
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
async def predict(
    file: UploadFile = File(...)
):

    contenido = await file.read()


    imagen = Image.open(
        io.BytesIO(contenido)
    ).convert("RGB")


    ancho_original, alto_original = imagen.size


    tensor = preprocess(
        imagen
    )


    outputs = session.run(
        None,
        {
            input_name:
            tensor
        }
    )


    predictions = outputs[0][0]


    predictions = predictions.transpose()


    boxes = []
    scores = []
    class_ids = []


    for pred in predictions:


        cx, cy, w, h = pred[:4]


        class_scores = pred[4:]


        class_id = np.argmax(
            class_scores
        )


        confidence = class_scores[class_id]


        if confidence > 0.5:


            # Convertir cx cy wh a x1 y1 x2 y2

            x1 = (
                cx - w / 2
            )

            y1 = (
                cy - h / 2
            )

            x2 = (
                cx + w / 2
            )

            y2 = (
                cy + h / 2
            )


            # Escalar a imagen original

            x1 *= ancho_original / 320
            x2 *= ancho_original / 320

            y1 *= alto_original / 320
            y2 *= alto_original / 320


            boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            )


            scores.append(
                float(confidence)
            )


            class_ids.append(
                int(class_id)
            )



    detecciones = []


    if len(boxes) > 0:


        keep = nms(
            boxes,
            scores
        )


        for i in keep:


            detecciones.append(

                {
                    "class":
                    classes[class_ids[i]],


                    "confidence":
                    round(
                        scores[i],
                        3
                    ),


                    "bbox":
                    [
                        round(
                            boxes[i][0],
                            1
                        ),
                        round(
                            boxes[i][1],
                            1
                        ),
                        round(
                            boxes[i][2],
                            1
                        ),
                        round(
                            boxes[i][3],
                            1
                        )
                    ]
                }

            )


    return {
        "detections":
        detecciones
    }