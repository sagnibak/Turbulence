# Project started on May 27, 2017 by Sagni(c)k Bhattacharya.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.optimizers import SGD
# from keras.activations import elu
from keras import initializers

# hyperparameters
epochs = 25000
batch_size = 50

# During hyperparameter optimization, this helps keep track of the version
# of code being run. Every time I make a change to a hyperparamter, I
# appropriately change the version name and number.
print("")
print("VERSION Layers2810.0")

# loading the datasets
print("Loading data...")
dataset = np.loadtxt("Turbulence_Training.csv", delimiter=",")
testset = np.loadtxt("Turbulence_Testing_Calm.csv", delimiter=",")  # testset is a palindrome!!! (doesn't matter)
new_inputs = np.loadtxt("Processed_Turbulence_Inputs_Big.csv", delimiter=',')
validation_inputs = np.loadtxt("Processed_Turbulence_Testing_Calm_1.csv", delimiter=',')

# separating inputs and outputs from the datasets
print("Processing the data...")
new_outputs = dataset[499:, 1:4]
validation_outputs = testset[499:-1, 1:4]

# Use the following ocmmented lines of code to modify the datasets if
# you want to change the number of input neurons of the MLP.

# inputs = dataset[:, 0:1]
# new_inputs = np.array([[]])
# for i in range(499, 37045):
#     new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]]))
# new_inputs = np.reshape(new_inputs, (36546, 500))
# np.savetxt("Processed_Turbulence_Inputs_Big.csv", new_inputs, delimiter=',')


# splitting the input and output data into training and testing data
X_train = new_inputs[5000:29000, :]
X_test = new_inputs[24000:, :]
Y_train = new_outputs[5000:29000, :]
Y_test = new_outputs[24000:, :]

# this is where the neural network is designed
print("Building the model...")
model = Sequential()
model.add(Dense(28, activation='tanh', input_dim=500))
model.add(Dropout(0.2))
model.add(Dense(10, activation='tanh'))
model.add(Dropout(0.2))
# model.add(Dense(5, activation='softplus')) # a brief experiment with 3 hidden layers...
# model.add(Dropout(0.2))                    # ...that didn't go well.
model.add(Dense(3, activation='softmax'))

# initializes the weights
initializers.TruncatedNormal(mean=1.0, stddev=0.05, seed=None)

# prints the structure of the neural network
model.summary()

# a failed experiment with sgd and momentum
# sgd = SGD(lr=1000.0, decay=0.0001, momentum=1.0e-1, nesterov=True)

# specifying the optimizer and the loss function
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# training the neural network!!!
print("Training the model...")
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),
          shuffle=True)

# evaluating the model's performance on data that it has never seen before
score = model.evaluate(X_test, Y_test, verbose=0)
score1 = model.evaluate(validation_inputs, validation_outputs, verbose=0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Calm Test loss:", score1[0])
print("Calm Test accuracy:", score1[1])

# This repository comes with a pre-trained model called 'Turbulence_Model.h5'
# You can use that model if you aren't interested in training your model. In that
# case, you should look at 'Turb_Test.py', where I load the pre-trained model
# and evaluate its performance.
# If you want to save a model that you've trained, then you can use the following
# line of code.

model.save('Turbulence_Model_2.h5')
