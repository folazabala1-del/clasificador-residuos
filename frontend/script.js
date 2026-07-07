const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const result = document.getElementById("result");
const status = document.getElementById("status");

const API_BASE = "https://clasificador-residuos-11h1.onrender.com";
const API_PREDICT = `${API_BASE}/predict`;
const API_HEALTH = `${API_BASE}/health`;

// --------------------------------------------------------------------
// "Calentamos" el servidor apenas carga la página, para que la primera
// detección real no pague el costo de que Render despierte el servicio.
// --------------------------------------------------------------------
async function calentarServidor() {
    status.textContent = "Conectando con el servidor (puede tardar si estaba dormido)...";
    status.className = "status";
    try {
        const inicio = performance.now();
        const res = await fetch(API_HEALTH);
        const ms = Math.round(performance.now() - inicio);
        if (res.ok) {
            status.textContent = `Servidor listo (${ms} ms)`;
            status.className = "status ok";
        } else {
            status.textContent = "El servidor respondió con un error.";
            status.className = "status error";
        }
    } catch (err) {
        status.textContent = "No se pudo conectar con el servidor.";
        status.className = "status error";
        console.error(err);
    }
}
calentarServidor();

// --------------------------------------------------------------------
// Cámara
// --------------------------------------------------------------------
navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
})
.then(stream => {
    video.srcObject = stream;
})
.catch(err => {
    console.error(err);
    status.textContent = "No se pudo acceder a la cámara.";
    status.className = "status error";
});

video.addEventListener("loadedmetadata", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
});

// --------------------------------------------------------------------
// Detección
// --------------------------------------------------------------------
let processing = false;

async function detect() {
    if (processing || video.videoWidth === 0) return;
    processing = true;

    // IMPORTANTE: capturamos el frame con el mismo ancho/alto de la
    // webcam (ej. 640x480), SIN forzarlo a un cuadrado de 320x320.
    // Deformar la imagen aquí (como antes) le da al modelo una forma
    // distinta a la que vio durante el entrenamiento, y eso baja mucho
    // la confianza de las detecciones. El backend se encarga de
    // redimensionar correctamente (manteniendo la proporción) antes de
    // pasarla al modelo.
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    tempCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const inicio = performance.now();

        try {
            const response = await fetch(API_PREDICT, {
                method: "POST",
                body: formData,
            });

            const latenciaMs = Math.round(performance.now() - inicio);

            if (!response.ok) {
                result.innerHTML = `⚠️ El servidor respondió con error (${response.status})`;
                processing = false;
                return;
            }

            const data = await response.json();

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (data.detections && data.detections.length) {
                result.innerHTML = "";

                // Ya NO se reescala: el backend devuelve las coordenadas
                // directamente en el tamaño del frame que se envió
                // (video.videoWidth x video.videoHeight), porque ahí
                // deshace el letterbox antes de responder.
                data.detections.forEach(det => {
                    const [x1, y1, x2, y2] = det.bbox;

                    ctx.strokeStyle = "lime";
                    ctx.lineWidth = 4;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    ctx.fillStyle = "lime";
                    ctx.font = "22px Arial";
                    ctx.fillText(
                        `${det.class} ${(det.confidence * 100).toFixed(1)}%`,
                        x1,
                        y1 - 10
                    );

                    result.innerHTML += `♻️ <b>${det.class}</b> (${(det.confidence * 100).toFixed(1)}%)<br>`;
                });
            } else {
                result.innerHTML = "Buscando residuo...";
            }

            result.innerHTML += `<span class="latencia">Latencia: ${latenciaMs} ms</span>`;

        } catch (error) {
            console.error(error);
            result.innerHTML = "⚠️ No se pudo contactar al servidor.";
        }

        processing = false;

    }, "image/jpeg", 0.8);
}

setInterval(detect, 300);