const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const changeImageBtn = document.getElementById('changeImageBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const resultsSection = document.getElementById('resultsSection');
const predictionResult = document.getElementById('predictionResult');
const confidenceValue = document.getElementById('confidenceValue');
const progressFill = document.getElementById('progressFill');
const resultMessage = document.getElementById('resultMessage');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

let selectedFile = null;

uploadBox.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;
    
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload PNG, JPG, JPEG, or BMP image');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        alert('File must be less than 16MB');
        return;
    }
    
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        analyzeBtn.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

changeImageBtn.addEventListener('click', resetUI);

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
            resetAnalyzeButton();
        }
    } catch (error) {
        alert('Error analyzing image. Please try again.');
        resetAnalyzeButton();
    }
});

function displayResults(data) {
    analyzeBtn.style.display = 'none';
    resultsSection.style.display = 'block';
    
    predictionResult.textContent = data.prediction;
    predictionResult.className = 'result-value ' + data.prediction.toLowerCase();
    
    confidenceValue.textContent = data.confidence + '%';
    progressFill.style.width = data.confidence + '%';
    
    if (data.prediction === 'Infected') {
        resultMessage.innerHTML = '<strong>⚠️ Malaria Parasite Detected</strong><br>The AI detected patterns consistent with malaria infection with ' + data.confidence + '% confidence. Requires confirmation through expert microscopy.';
    } else {
        resultMessage.innerHTML = '<strong>✓ No Malaria Detected</strong><br>The AI did not detect malaria parasites with ' + data.confidence + '% confidence. Verify with a medical professional if symptoms persist.';
    }
}

analyzeAnotherBtn.addEventListener('click', resetUI);

function resetUI() {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    analyzeBtn.style.display = 'none';
    resultsSection.style.display = 'none';
    resetAnalyzeButton();
}

function resetAnalyzeButton() {
    analyzeBtn.disabled = false;
    btnText.style.display = 'inline';
    btnLoader.style.display = 'none';
}