import keras.models
from keras.layers import GlobalMaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.applications import EfficientNetB4
from keras import Sequential
import math
import urllib
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
import numpy as np
from fastapi import FastAPI
from keras.metrics import mean_absolute_error
from keras import backend as K
import tensorflow as tf
from uvicorn import run
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
import httpx
import json

img_size = 380

def load_and_preprocess_image(url):
    image_path, _ = urllib.request.urlretrieve(url)
    img = image.load_img(image_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
def predict_image(url):
    img_array = load_and_preprocess_image(url)
    prediction = model.predict(img_array)
    return prediction[0][0]

def load_and_preprocess_image_to_verify(url):
    image_path, _ = urllib.request.urlretrieve(url)
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = vgg_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def verify_image(url):
    img_array = load_and_preprocess_image_to_verify(url)
    print("Yo")
    prediction = model_hands.predict(img_array)
    return np.argmax(prediction[0]).item()

model = tf.keras.models.load_model('meu_modelo.h5')
model.load_weights('meus_pesos.h5')

app = FastAPI()

model_hands = keras.models.load_model("hands2.h5")
model_hands.load_weights("model_weights2.h5")

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
    return {"message": "Welcome to the Boneage AI!"}

@app.post("/net/image/prediction")

async def testando(request: Request):
    if request.body:
        request_body = await request.body()
        body_dict = json.loads(request_body)
        image_link = body_dict.get("image_link")
        if image_link:
            if image_link == "":
                return {"message": "No image link provided"}
            prediction = predict_image(image_link)
            print("Prediction:", prediction / 12)

            return {
                "prediction": round(prediction)
            }

@app.post("/verify")

async def verify_image_api(request: Request):
    try:
        if request.body:
            body_dict = await request.json()
            image_link = body_dict.get("image_link")
            if image_link:
                if image_link == "":
                    return {"message": "No image link provided"}
                prediction = verify_image(image_link)
                print("Prediction:", prediction)

                return {
                    "prediction": not bool(prediction)
                }
    except Exception as e:
        print("(com a voz do faustao) errrrrrrrrou:")
        print(e)

@app.put("/weight")

async def swap_weights(request: Request):
    try:
        if not request.body:
            raise Exception("Missing body")
        body_dict = await request.json()
        file_link = body_dict.get("file_link")
        if not file_link:
            raise Exception("Missing 'file_link' parameter") 


        async with httpx.AsyncClient() as client:
            response = await client.get(file_link)

            if response.status_code == 200:
                file_name = "temp_weights"
                file_extension = file_link.split('.')[-1]  
                file_name_with_extension = f"{file_name}.{file_extension}"

                if file_extension != 'h5':
                    raise Exception("Invalid file format. Only .h5 files are allowed.") 

                file_path = os.path.join(os.getcwd(), file_name_with_extension)
                with open(file_path, 'wb') as file:
                    file.write(response.content)

                #Checa a compatibilidade
                try:
                    model.load_weights(file_name_with_extension)
                    print("Model and weights are compatible.")
                    os.replace(file_name_with_extension, "bone_age_weights.h5")
                except Exception as e:
                    print(e)
                    model.load_weights("bone_age_weights.h5")
                    os.remove(file_name_with_extension)
                    return JSONResponse(content={"error": "Model and weights are not compatible."}, status_code=422)

                return {"message": "Weights swaped"}
            else:
                raise Exception(f"Failed to download file: {response.text}")
    except Exception as e:
        print("(com a voz do faustao) errrrrrrrrou:")
        print(e)
        return JSONResponse(content={"error": "Erro ao trocar pesos"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5101))
    run(app, host="0.0.0.0", port=port)
