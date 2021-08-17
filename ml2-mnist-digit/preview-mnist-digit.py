# example of loading the mnist dataset
import pickle
import sys

from keras.datasets import mnist
from matplotlib import pyplot

# load dataset from provided keras dataset (either from aws or gcp public endpoint)
# the file is in the form of npz
(trainX, trainY), (testX, testY) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

# generate first few images as sample
for i in range(9):
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
	# save the image
	pyplot.savefig('sample/preview%d.png' % i)

# preview image
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))

# show the figure
pyplot.show()
