const API_URL = 
"https://clasificador-residuos-11h1.onrender.com/predict";


const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const ctx = canvas.getContext("2d");


const status = document.getElementById("status");
const prediction = document.getElementById("prediction");


// =======================
// Activar cámara
// =======================

navigator.mediaDevices.getUserMedia({
    video:true
})
.then(stream => {

    video.srcObject = stream;

    status.innerHTML = "Cámara activa";

})
.catch(error => {

    console.error(error);

    status.innerHTML =
    "No se pudo acceder a la cámara";

});



// =======================
// Detectar residuos
// =======================


async function detectar(){


    if(video.readyState !== 4)
        return;



    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;



    // Capturar frame

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );



    canvas.toBlob(async(blob)=>{


        const formData = new FormData();

        formData.append(
            "file",
            blob,
            "camera.jpg"
        );



        try{


            const response =
            await fetch(
                API_URL,
                {
                    method:"POST",
                    body:formData
                }
            );



            const data =
            await response.json();



            dibujarResultados(data);



        }
        catch(error){

            console.error(error);

        }


    }, "image/jpeg",0.8);


}



// =======================
// Dibujar cajas
// =======================

function dibujarResultados(data){


    // limpiar canvas

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    if(data.detections.length === 0){

        prediction.innerHTML =
        "No detectado";

        return;

    }



    data.detections.forEach(det=>{


        const [
            x1,
            y1,
            x2,
            y2
        ] = det.bbox;



        ctx.strokeStyle="red";
        ctx.lineWidth=3;


        ctx.strokeRect(
            x1,
            y1,
            x2-x1,
            y2-y1
        );


        ctx.fillStyle="red";
        ctx.font="20px Arial";


        ctx.fillText(
            `${det.class} ${(det.confidence*100).toFixed(1)}%`,
            x1,
            y1-10
        );


        prediction.innerHTML =
        `
        Residuo:
        ${det.class}
        <br>
        Confianza:
        ${(det.confidence*100).toFixed(1)}%
        `;


    });


}



// Ejecutar cada 500ms

setInterval(
    detectar,
    3000
);