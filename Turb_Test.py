import numpy as np
from keras.models import load_model

print("")
print("VERSION Valid.Test")

print("")
print("Loading data...")
testtset = np.loadtxt("Turbulence_Testing_Turbulent.csv", delimiter=",")  # testtset is a palindrome too!!!

print("Processing the data...")
validation_outputs_turb = testtset[499:-1, 1:4]
validation_inputs_turb = np.loadtxt("Processed_Turbulence_Testing_Turbulent.csv", delimiter=',')

print("Loading Model...")
model = load_model('Turbulence_Model.h5')

score2 = model.evaluate(validation_inputs_turb, validation_outputs_turb, verbose=0)

print("Turbulent Test loss:", score2[0])
print("Turbulent Test accuracy:", score2[1])
