# This is a neural network to determine if an airplane is flying through very turbulent,
# moderately turbulent, or calm weather. Initially, my plan was to use an LSTM, but that
# meant processing a lot of data, so I went for an MLP instead, which did not need me to
# process the data as much.
#
# I prepared the Turbulence_Training.csv dataset myself, by recording accelerometer data
# from my phone while flying from New Delhi to San Francisco. I made quite a few record-
# ings, but chose a three-minute-long file with a mix of all three kinds of data, and then
# (arguably foolishly) manually labelled the 37000+ data points as Turb, MTurb or Calm.
#
# epochs = 1000, batch_size = 100,
# 'adam', 'categorical_crossentropy',
#
# model.add(Dense(20, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.712320870909
# Test accuracy: 0.666692414645
#
# epochs = 5000
#
# Test loss: 0.600918423797
# Test accuracy: 0.714969874865
#
# model.add(Dense(30, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.586811004088
# Test accuracy: 0.727792368299 (actually maxed around 0.7347)
#
# model.add(Dense(30, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: {aborted after remaining stuck for 500 epochs}
# Test accuracy: {aborted after staying stuck for 500 epochs}
#
#
# model.add(Dense(32, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.603847062507
# Test accuracy: 0.705314382821 (actually maxed around 0.7225)
#
#
# model.add(Dense(32, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.599734134229
# Test accuracy: 0.71612853391 (actually maxed around 0.7237)
#
# model.add(Dense(28, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.5797
# Test accuracy: 0.7404
#
# model.add(Dense(28, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(5, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: {not recorded}
# Test accuracy: 0.69<accuracy<0.70
#
# model.add(Dense(28, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: {aborted after staying stuck for 800 epochs}
# Test accuracy: {aborted after staying stuck for 800 epochs}
#
# model.add(Dense(28, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: {aborted after staying stuck for 1200 epochs}
# Test accuracy: {aborted after staying stuck for 1200 epochs}
#
# model.add(Dense(28, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: {aborted after staying stuck for 500 epochs}
# Test accuracy: {aborted after staying stuck for 500 epochs}
#
# model = Sequential()
# model.add(Dense(28, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='softplus'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.615324148614
# Test accuracy: 0.707399969102
# Calm Test loss: 6.19425047046
# Calm Test accuracy: 0.215362997658 # OVERFITTING!!!
#
#
# model.add(Dense(28, activation='sigmoid', input_dim=100))
# model.add(Dropout(0.4))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.4))
# model.add(Dense(4, activation='softplus'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# {this model learnt nothing in 5000 epochs due to the enormous amounts of dropout}
#
# I think that the reason that the model is not training properly is that the input layer
# is only 100 units wide, scanning over a little than 0.5 seconds of data at once. So, I
# am now modifying the data to allow the MLP to have 500 neurons in the input layer. This
# will allow the network to scan 2.5 seconds into the past, which should increase accuracy.
#
# As of now, it looks like there is something wrong with the 500-neuron architecture. I
# tried different 2- and 3-layer configurations with ReLU activation, all of which failed
# to converge in 200-500 epochs. Then I tried a sigmoid architecture as well (the one that
# worked the best with 100 input neurons), which also failed to converge. Now I want to
# play around a little with the optimizer.
#
#
# --------------------------------------------------------------------------------------------------------------------
# epochs = 25000
# batch_size = 100
#
# print("")
# print("VERSION TanhLM.1")
#
# print("Loading data...")
# dataset = np.loadtxt("Turbulence_Training.csv", delimiter=",")  # load the dataset
# # testset = np.loadtxt("Turbulence_Testing_Calm.csv", delimiter=",")  # testset is a palindrome!!!
#
# print("Processing the data...")
# inputs = dataset[:, 0:1]
# new_outputs = dataset[499:, 1:4]
# # validation_outputs = testset[99:-1, 1:4]
#
# # new_inputs = np.array([[]])
# # for i in range(499, 37045):
# #     new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]]))
# # new_inputs = np.reshape(new_inputs, (36546, 500))
# # np.savetxt("Processed_Turbulence_Inputs_Big.csv", new_inputs, delimiter=',')
#
# new_inputs = np.loadtxt("Processed_Turbulence_Inputs_Big.csv", delimiter=',')
# # validation_inputs = np.loadtxt("Processed_Turbulence_Testing_Calm.csv", delimiter=',')
#
# X_train = new_inputs[5000:29000, :]
# X_test = new_inputs[24000:, :]
# Y_train = new_outputs[5000:29000, :]
# Y_test = new_outputs[24000:, :]
#
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)
#
# print("Building the model...")
# model = Sequential()
# model.add(Dense(28, activation='tanh', input_dim=500))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='tanh'))
# model.add(Dropout(0.2))
# # model.add(Dense(5, activation='softplus'))
# # model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# model.summary()
#
# sgd = SGD(lr=1000.0, decay=0.001, momentum=1.0e-1, nesterov=False)
# model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
#
# Epoch 4438/25000
# 24000/24000 [==============================] - 0s - loss: 0.5181 - acc: 0.7141 - val_loss: 0.4618 - val_acc: 0.7603
#
# --------------------------------------------------------------------------------------------------------------------
#
# Test loss: 0.43035825716
# Test accuracy: 0.790876241158 (max around 0.7937)
#
# model.add(Dense(32, activation='tanh', input_dim=500))
# model.add(Dropout(0.2))
# model.add(Dense(11, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.440905945767
# Test accuracy: 0.783463519212 (max around 0.7850)
#
#
# model.add(Dense(28, activation='tanh', input_dim=500))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
#  model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#
# Test loss: 0.344701589349
# Test accuracy: 0.842101072326
#
#
# model.add(Dense(24, activation='tanh', input_dim=500))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# Test loss: 0.353813521501
# Test accuracy: 0.837770340185
# Something to be noted about this training run is that the Test Accuracy was actually lower
# than the accuracy on the training data. Possibly, this is a sign of overfitting. So I am
# increasing both the dropouts from 0.2 to 0.25. Let's see what happens.
#
# Original results were lost, but accuracy on the testing data maxed around 0.8370. One thing
# to be noted is that it ket bouncing in the 0.80-0.83 ballpark for several thousand epochs.
#
# model.add(Dense(20, activation='tanh', input_dim=500))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
#
# Test loss: 0.416595364207
# Test accuracy: 0.800414479066 (max around 0.8270)
# This time too, the accuracy kept bouncing around in the 0.80-0.83 ballpark for the last 5000
# epochs or so.
#
#
# model = Sequential()
# model.add(Dense(28, activation='tanh', input_dim=500))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
#
# initializers.TruncatedNormal(mean=1.0, stddev=0.05, seed=None)
#
# Test loss: 0.352511857044
# Test accuracy: 0.83877996005 (max around 0.8404)
#
#
# batch_size = 80
#
#
# Test loss: 0.35624565698
# Test accuracy: 0.838036028407 (max around 0.8442)
#
#
# batch_size = 70
#
# Test loss: 0.354777779751
# Test accuracy: 0.839205062011 (max around 0.8457)
#
#
# -----------------------------------------------------------------------------------------------
# ***THE BEGINNING OF THE END***
# -----------------------------------------------------------------------------------------------
# batch_size = 60
#
# I also decided to finally process the Turbulence_Testing_Calm dataset for the
# 500-input-neuron MLP, so now there is FINALLY some validation data that the model
# has really NEVER seen before. If the model, which is consistently classifying 5
# out of 6 examples correctly, performs well on this dataset (unlike the 100-input-neuron
# MLP that I had designed earlier), then I will finally put this code on GitHub and
# call it a day. *fingers crossed*
#
# Test loss: 0.3521712085
# Test accuracy: 0.837345238072
# Calm Test loss: 0.0760556243008
# Calm Test accuracy: 1.0
#
# I CAN'T BELIEVE THAT THIS IS REAL!!! Okay. So the neural network does know the first thing
# about turbulence. But before I upload this code, I want to make sure that it can correctly
# identify turbulent motion as well. Then I will throw in the towel.
#
# Turbulent Test loss: 0.408611278705
# Turbulent Test accuracy: 0.80676939629
#
# I am impressed with the performance of this MLP. Even though it did not classify turbulent
# weather as accurately as it classified calm weather, it performed pretty well, proving that
# it is generalizing. It's also worth noting that turbulent data is a little hard to differentiate
# from moderately turbulent weather, even for a human, so this MLP performed pretty well.
#
# ---------------------------------------------------------------------------------------------------
#
# Today, on June 15, 2017, I, Sagni(c)k Bhattacharya, am committing this code on GitHub.
#
# Project started on May 27, 2017 by Sagni(c)k Bhattacharya.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.optimizers import SGD
# from keras.activations import elu
from keras import initializers

# hyperparameters
epochs = 25000
batch_size = 60

# During hyperparameter optimization, this helps keep track of the version
# of code being run. Every time I make a change to a hyperparamter, I
# appropriately change the version name and number.
print("")
print("VERSION Valid.2")

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
model.add(Dense(8, activation='tanh'))
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

# model.save('Turbulence_Model.h5')
