from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# For version handling we are using tf_serving
# 1. postman will call fastapi
# 2. FastAPI will do numpy conversion and all, then will call the tf_serving
# 3. tf_serving will do the prediction 
app = FastAPI()

model = tf.keras.models.load_model("../saved/versions/1/potato_disease_model.keras")
class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app.get("/ping") # Specification of endpoint
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    
    return image

@app.post("/predict") # Specification of endpoint
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    prediction = model.predict(img_batch)
    predicted_class = class_name[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

'''if __name__ =="__main__":
    uvicorn.run(app, host='localhost', port=8000)'''