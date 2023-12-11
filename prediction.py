import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

def getPrediction(file_path):
        
    #Load model
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    my_model = model_from_json(loaded_model_json)
    my_model.load_weights('model_cnnv2.h5')
    
    imagenes_array_train = []
    imagen_array = np.asarray(Image.open(file_path).convert("RGB").resize((176, 208)))       
    
    # Añadir el array NumPy a la lista
    imagenes_array_train.append(imagen_array)
    img = np.stack(imagenes_array_train)
    pred = my_model.predict(img) #Predict                    
    y_pred = [np.argmax(pred)][0]
    
    if y_pred == 0:
        return 'NonDemented'
    elif y_pred == 1:
        return 'VeryMildDemented'
    elif y_pred == 2:
        return 'MildDemented'
    elif y_pred == 3:
        return 'ModerateDemented'
    
