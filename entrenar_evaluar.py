import sys
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


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


# Curvas de aprendizaje para análisis
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # Guardamos el plot para analizar
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot_' + '.png')
    pyplot.close()


def run_test():
    # Instanciamos el modelo
    model = define_model()

    # Creamos un generador de imagen
    datagen = ImageDataGenerator(featurewise_center=True)

    # Ponemos el foco de entrenamiento en el centro de las imagenes
    datagen.mean = [123.68, 116.779, 103.939]

    # Generador para entrenar
    train_it = datagen.flow_from_directory('datasets/apto_vs_no_apto/entrenamiento',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))
    # Generador para validar
    validate_it = datagen.flow_from_directory('datasets/apto_vs_no_apto/validacion/',
                                          class_mode='binary', batch_size=64, target_size=(224, 224))
    # Entrenamos el modelo
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=validate_it, validation_steps=len(validate_it), epochs=10, verbose=1)
    # Evaluamos del modelo e imprimimos por pantalla
    _, acc = model.evaluate_generator(validate_it, steps=len(validate_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    # Imprimimos curvas de apendizaje del proceso
    summarize_diagnostics(history)

# Main
run_test()