# code from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# first neural network with keras tutorial
from pathlib import Path

from keras.engine.saving import model_from_json
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset as a matrix of numbers
dataset = loadtxt('dataset/pima-indians-diabetes.csv', delimiter=',')
# split into input (X: col 1-8) and output (y: col 9) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

out_path = 'model'
my_file = Path(out_path+"/model.json")
if my_file.is_file():
    json_file = open(out_path+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(out_path+'/model.h5')
    print("Loaded model from disk")
else:
    # define the keras model
    model = Sequential()
    # the activation function is
    # responsible for transforming the summed weighted input from the node
    # into the activation of the node or output for that input.
    # activation
    #   - Rectified Linear Unit (ReLU): output the input directly if it is positive, otherwise, it will output zero
    #   to help models to learn faster and perform better.
    #   - sigmoid: The input to the function is transformed into a value between 0.0 and 1.0
    #   - hyperbolic tangent (tanh): The input to the function is transformed into a values between -1.0 and 1.0
    # 1st (input) layer with 12 nodes
    # receive array of 8 and return array of 12 members
    model.add(Dense(12, input_dim=8, activation='relu'))
    # 2nd (hidden) layer with 8 nodes
    # receive array of previous layer and return array of 8 members
    model.add(Dense(8, activation='relu'))
    # 3rd (output) layer with 1 node
    # receive array of previous layer and return array of 1 member
    model.add(Dense(1, activation='sigmoid'))

# compile the keras model using backend tool.
# The backend automatically chooses the best way to represent the network for training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model - fit the keras model on the dataset
# epochs is number times that the learning algorithm will work through the entire training dataset
# batch_size is number of samples to work through before updating the internal model parameters
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# serialize model to JSON
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model.h5")
print("Saved model to disk")