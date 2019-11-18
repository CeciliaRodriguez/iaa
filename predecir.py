import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

testPath = os.path.join(os.getcwd(), 'test')

#Cargamos la imagen en memoria y la ajustamos para evaluar
def load_image(filename):
    # Carga de la imagen a memoria
    img = load_img(filename, target_size=(224, 224))
    # La convertimos a array para poder trabajarla
    img = img_to_array(img)
    # Hacemos reshape sobre 3 canales
    img = img.reshape(1, 224, 224, 3)
    # Por último centramos la imagen
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# load an image and predict the class
def run_example():
    # Cargamos el modelo entrenado
    model = load_model('final_model.h5')
    print('Clasificación: Apto [0] No apto [1]')
    print('Salida: Archivo -> Clasificación')
    print('--------------------------------')
    for filename in os.listdir(testPath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Cargamos la imagen para testear
            img = load_image(testPath+'/'+filename)
            # Realizamos la predicción
            result = model.predict(img)
            print(filename + ' -> %f' % result[0])
        else:
            continue

# entry point, run the example
run_example()