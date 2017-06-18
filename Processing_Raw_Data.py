# Run this code BEFORE running either of the other Python files, to generate the processed
# datasets that they will need in order to run. Ideally, you will have to run this file only once.

import numpy as np


# Code to process 'Turbulence_Testing_Calm.csv'...
# ...and generate 'Processed_Turbulence_Testing_Calm_1.csv'

# load the dataset, i.e., convert the .csv file into a Numpy array called `dataset`
print("Loading data...")
dataset = np.loadtxt("Turbulence_Training.csv", delimiter=",")

# isolate the inputs from the outputs by slicing the Numpy array `dataset`
print("Processing the data...")
inputs = dataset[:, 0:1]

# This is the complex part where the data is rearranged. Refer to my Instructable
# here ______________________ for details about how this code works.
new_inputs = np.array([[]]) # initialize an empty Numpy array

for i in range(499, 37045):
    new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]])) # add numbers in the right order to the empty array

new_inputs = np.reshape(new_inputs, (36546, 500)) # change the shape of new_inputs from 1*18273000 to 36546*500
np.savetxt("Processed_Turbulence_Inputs_Big.csv", new_inputs, delimiter=',')

# looking at
#   1) whether the file loads, and
#   2) the shape of the array
# is a quick way to verify that the black box above generated the right file.
a = np.loadtxt("Processed_Turbulence_Inputs_Big.csv", delimiter=',')
print(a.shape)

# The other two blocks work the same way, so I have not put comments on them.

# Code to process 'Turbulence_Testing_Calm.csv'...
# ...and generate 'Processed_Turbulence_Testing_Calm_1.csv'
print("Loading data...")
dataset = np.loadtxt("Turbulence_Testing_Calm.csv", delimiter=",")

print("Processing the data...")
inputs = dataset[:, 0:1]

new_inputs = np.array([[]])
for i in range(499, 10774):
    new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]]))
new_inputs = np.reshape(new_inputs, (10275, 500))
np.savetxt("Processed_Turbulence_Testing_Calm_1.csv", new_inputs, delimiter=',')

a = np.loadtxt("Processed_Turbulence_Testing_Calm_1.csv", delimiter=',')
print(a.shape)


# Code to process 'Turbulence_Testing_Turbulent.csv'...
# ...and generte 'Processed_Turbulence_Testing_Turbulent.csv'
print("Loading data...")
dataset = np.loadtxt("Turbulence_Testing_Turbulent.csv", delimiter=",")

print("Processing the data...")
inputs = dataset[:, 0:1]

new_inputs = np.array([[]])
for i in range(499, 17113):
    new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]]))
new_inputs = np.reshape(new_inputs, (16614, 500))
np.savetxt("Processed_Turbulence_Testing_Turbulent.csv", new_inputs, delimiter=',')

a = np.loadtxt("Processed_Turbulence_Testing_Turbulent.csv", delimiter=',')
print(a.shape)
