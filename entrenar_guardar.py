from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import os, glob, shutil

def moveDatasetFromTo(source_dir, source_sub_dir, dest_dir):
    source_dir = os.path.join(os.getcwd(), source_dir)
    dest_dir = os.path.join(os.getcwd(), '/'.join([dest_dir, source_sub_dir]))
    for subdir in ['/'.join(['entrenamiento', source_sub_dir]), '/'.join(['validacion', source_sub_dir])]:
        source_dir = os.path.join(source_dir, subdir)
        for filename in glob.glob(os.path.join(source_dir, '*.*')):
            shutil.copy(filename, dest_dir)

# Copiamos de las carpetas de entrenamiento y validacion
def prepare_final_dataset():
    moveDatasetFromTo('datasets/apto_vs_no_apto', 'apto', 'datasets/apto_vs_no_apto_entrenamiento_completo/')
    moveDatasetFromTo('datasets/apto_vs_no_apto', 'no_apto', 'datasets/apto_vs_no_apto_entrenamiento_completo/')

# Definición del la red convulcional
def define_model():
    # Creamos el modelo
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # Marcamos las capas como no entrenables
    for layer in model.layers:
        layer.trainable = False
    # Agregamos capas para la clasificación
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # Creamos un nuevo modelo en base a VGG y la capa de clasificación
    model = Model(inputs=model.inputs, outputs=output)
    # Compilamos el modelo
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Evaluamos el modelo y lo guardamos
def run_save():
    # Instanciamos el modelo
    model = define_model()
    # Preparar el dataset final
    prepare_final_dataset()
    # Creamos un generador de imagen
    datagen = ImageDataGenerator(featurewise_center=True)
    # Ponemos el foco de entrenamiento en el centro de las imagenes
    datagen.mean = [123.68, 116.779, 103.939]
    # Preparamos el iterador con imagenes del dataset para entrenar la red final
    train_it = datagen.flow_from_directory('datasets/apto_vs_no_apto_entrenamiento_completo/',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))
    # Entrenamos el modelo en sí
    model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
    # Guardamos el modelo
    model.save('final_model.h5')

# Main
run_save()