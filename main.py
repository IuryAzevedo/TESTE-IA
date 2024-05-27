import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uvicorn import run
from PIL import Image
import os

# Definindo o modelo em PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*6*6, 256)
        self.fc2 = nn.Linear(256*6*6, 256)
        self.dropout = nn.Dropout(0.3)
        self.gender_out = nn.Linear(256, 1)
        self.age_out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = self.flatten(x)
        x1 = self.dropout(torch.relu(self.fc1(x)))
        x2 = self.dropout(torch.relu(self.fc2(x)))
        gender = torch.sigmoid(self.gender_out(x1)).squeeze(1)
        age = self.age_out(x2).squeeze(1)
        return gender, age

# Verificar se CUDA está disponível e definir o dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instanciar o modelo e carregar os pesos
model = CNN()
model.load_state_dict(torch.load('meu_modelo.pth', map_location=device))
model.to(device)
model.eval()

# Função para carregar e preprocessar a imagem
def load_and_preprocess_image(file):
    img = Image.open(file).convert('L')
    img = img.resize((128, 128))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0)  # Adiciona uma dimensão de batch
    return img_tensor.to(device)

# Função de predição
def predict_image(file):
    img_tensor = load_and_preprocess_image(file)
    with torch.no_grad():
        pred_gender, pred_age = model(img_tensor)
        print(f'Raw prediction - Gender: {pred_gender.item()}, Age: {pred_age.item()}')
        pred_gender = 'Male' if round(pred_gender.item()) == 0 else 'Female'
        pred_age = round(pred_age.item())
    return pred_gender, pred_age

# Configuração da API FastAPI
app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Age Gender AI!"}

@app.post("/net/image/prediction")
async def testando(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(content={"message": "Invalid image format"}, status_code=400)

        # Salvar a imagem temporariamente
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        pred_gender, pred_age = predict_image(file_location)

        # Remover a imagem temporária
        os.remove(file_location)

        return {
            "prediction": {
                "gender": pred_gender,
                "age": pred_age
            }
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5101))
    run(app, host="0.0.0.0", port=port)
