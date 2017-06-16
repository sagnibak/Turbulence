import numpy as np


# Code to process 'Turbulence_Testing_Calm.csv'...
# ...and generate 'Processed_Turbulence_Testing_Calm_1.csv'
print("Loading data...")
dataset = np.loadtxt("Turbulence_Training.csv", delimiter=",")  # load the dataset

print("Processing the data...")
inputs = dataset[:, 0:1]

new_inputs = np.array([[]])
for i in range(499, 37045):
    new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]]))
new_inputs = np.reshape(new_inputs, (36546, 500))
np.savetxt("Processed_Turbulence_Inputs_Big.csv", new_inputs, delimiter=',')

a = np.loadtxt("Processed_Turbulence_Inputs_Big.csv", delimiter=',')
print(a.shape)


# Code to process 'Turbulence_Testing_Calm.csv'...
# ...and generate 'Processed_Turbulence_Testing_Calm_1.csv'
print("Loading data...")
dataset = np.loadtxt("Turbulence_Testing_Calm.csv", delimiter=",")  # load the dataset

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
dataset = np.loadtxt("Turbulence_Testing_Turbulent.csv", delimiter=",")  # load the dataset

print("Processing the data...")
inputs = dataset[:, 0:1]

new_inputs = np.array([[]])
for i in range(499, 17113):
    new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]]))
new_inputs = np.reshape(new_inputs, (16614, 500))
np.savetxt("Processed_Turbulence_Testing_Turbulent.csv", new_inputs, delimiter=',')

a = np.loadtxt("Processed_Turbulence_Testing_Turbulent.csv", delimiter=',')
print(a.shape)
