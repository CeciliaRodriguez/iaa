# Imprimir fotos del dataset elegido por argumento
import os, sys
from matplotlib import pyplot
from matplotlib.image import imread


# define location of dataset
folder = os.path.join(rootDir, '/'.join([setDir, classDir]))
# plot first few images
files = os.listdir(folder)
for i in enumerate(files):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'drunk_' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)

# Mostramos imagen
pyplot.show()