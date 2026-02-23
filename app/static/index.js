// Afficher l'image chargée 
document.getElementById("fileInput").addEventListener("change", function() { 
    const file = this.files[0]; 
    if (!file) return; 

    const preview = document.getElementById("preview"); 
    preview.src = URL.createObjectURL(file); preview.style.display = "block";
});

//python -m http.server 5500 lacer server frontend
async function sendImage() {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Choisis une image !");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("result").innerText =
        "Le plat est : " + data.prediction;
}
