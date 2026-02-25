const fileInput = document.getElementById("fileInput");
const fileNameDisplay = document.getElementById("fileName");
const preview = document.getElementById("preview");
const gradcamImg = document.getElementById("gradcam");
const predictionList = document.getElementById("prediction-list");
const predictBtn = document.getElementById("predictBtn");

// Affichage du nom + preview dès upload
fileInput.addEventListener("change", () => {
    if(fileInput.files.length){
        const file = fileInput.files[0];
        fileNameDisplay.textContent = "📷 " + file.name;
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    }
});

predictBtn.addEventListener("click", sendImage);

async function sendImage(){

    if(!fileInput.files.length){
        alert("Choisis une image !");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try{
        const response = await fetch("http://127.0.0.1:8000/predict",{
            method:"POST",
            body:formData
        });

        if(!response.ok) throw new Error("Erreur serveur");

        const data = await response.json();

        predictionList.innerHTML = "";

        for(let i=0;i<Math.min(5,data.prediction.length);i++){

            const prob = (data.class_prob[i]*100).toFixed(2);

            const item = document.createElement("div");
            item.classList.add("prediction-item");

            item.innerHTML = `
                <div class="prediction-header">
                    <span>${data.prediction[i]}</span>
                    <span>${prob}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            `;

            predictionList.appendChild(item);

            // animation progressive
            setTimeout(()=>{
                item.querySelector(".progress-fill").style.width = prob+"%";
            },100);
        }

        gradcamImg.src = "data:image/png;base64," + data.gradcam;
        gradcamImg.style.display = "block";

    }catch(error){
        alert("Erreur lors de la communication avec l'API.");
        console.error(error);
    }
}