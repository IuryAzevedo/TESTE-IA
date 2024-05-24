from pyexpat import model
import keras.models
from keras.layers import GlobalMaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.applications import EfficientNetB4
from keras import Sequential
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
import numpy as np
from fastapi import FastAPI, File, UploadFile
from keras.metrics import mean_absolute_error
from keras import backend as K
import tensorflow as tf
from uvicorn import run
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Carregar o modelo treinado

#custom_objects = {"batch_shape": (1, 128, 128, 1)}  # Defina a especificação de batch_shape como um objeto personalizado
model = tf.keras.models.load_model('meu_modelo.h5')
model.save_weights('my_weights.weights.h5')
img_size = 128

def load_and_preprocess_image(file):
    img = Image.open(file).convert('L')  # Abre a imagem e converte para escala de cinza
    img = img.resize((img_size, img_size))  # Redimensiona para o tamanho desejado
    img_array = np.array(img)  # Converte a imagem em um array numpy
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão para corresponder ao formato desejado (altura, largura, canal)
    img_array = preprocess_input(img_array)  # Pré-processamento adicional, se necessário
    return img_array

def predict_image(file):
    img_array = load_and_preprocess_image(file)
    prediction = model.predict(img_array)
    return prediction

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

        prediction = predict_image(file_location)

        # Remover a imagem temporária
        os.remove(file_location)

        gender = prediction[0][0]
        age = prediction[1][0]
        print(gender, age)
        
        new_gender = int(gender[0])
        new_age = round(age[0])
    

        return {
            "prediction": {
                "gender": "Male" if new_gender == 0 else "Female",
                "age": new_age
            }
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5101))
    run(app, host="0.0.0.0", port=port)