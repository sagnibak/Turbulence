# Turbulence
###### *__Project started on May 27, 2017 by Sagni(c)k Bhattacharya.__*
This repository contains code to make a neural network that determines
 if an aircraft is flying through very turbulent, somewhat turbulent, 
 or calm weather based on accelerometer readings. This also includes 
 datasets and unlabeled data that requires processing to be used as 
 datasets. The neural networks are written in Python, using Keras 
 with TensorFlow backend.
 
This is a neural network to determine if an airplane is flying 
through very turbulent, moderately turbulent, or calm weather. 
Initially, my plan was to use an LSTM, but that meant processing a 
lot of data, so I went for an MLP instead, which did not need me 
to process the data as much.

I prepared the Turbulence_Training.csv dataset myself, by recording 
accelerometer data from my phone while flying from New Delhi to 
San Francisco. I made quite a few recordings, but chose a 
three-minute-long file with a mix of all three kinds of data, 
and then (arguably foolishly) manually labelled the 37000+ data 
points as Turb, MTurb or Calm (for  turbulent, moderately turbulent, 
and calm weather respectively).

I finished working on this repository on June 20, 2017. Since then,
I have learned a lot more about neural networks, and I have even done
research on neural networks. I have come to realize that although the
code here *works*, it is very, I mean *very* noobish. The architecture
of the neural network (a vanilla multilayer perceptron) is ancient, 
and the code is not good a model for beginners to follow. I am, however, 
letting this repo live because a) I worked hard to make it, and b) 
for all the noobs out there: you're not alone :smiley:

## The Files
**Turbulence_Training.csv:** This dataset contains the data 
that our model is trained and tested with during training.

**Turbulence_Testing_Calm.csv:** This dataset contains examples of 
 flying through calm conditions only. This is one of the two
 datasets used to validate the neural network. This is data that
 the neural network has never seen before. Hence, we can look for
 things like overfitting, and compare different models during hyperparameter
 optimization.
 
**Turbulence_Testing_Turbulent.csv:** This dataset is identical to the
`Turbulence_Testing_Calm.csv` dataset in terms of function. We use
it to validate the model after training, during hyperparameter 
optimization. This dataset contains examples of flight through turbulent
conditions only.

**Processing_Raw_Data.py:** This is a Python file that
should be executed once. It will process the data in 
`Turbulence_Testing_Calm.csv`, `Turbulence_Testing_Turbulent.csv`,
and `Turbulence_Training.csv` to generate three `.csv` files, which
have the data in the right format for the neural network to
work with a Multi-Layer Perceptron with 500 input neurons. Check
out my Instructable ______________ to learn how this file
works. *WARNING: The `.csv` files generated take up over 756 MB (0.74 GB)
of space in all. (Once you get deep into deep learning, you'll
realize that these are some of the smallest datasets you'll ever come across.)*

**Turbulence.py:** This is the heart of this repository. This 
contains the neural network that we design and train. This has 
bee discussed in detail in my Instructable _________.

**Turbulence_Model.h5:** This is the first of three pre-trained
models included in this repository. This will let you study the neural
network without having to train it first (training can take very
long if you don't have a good GPU). The code in this one is from
[training run 22](http://github.com/sagnibak/Turbulence#run-22).

**Turbulence_Model_1.h5:** This is the second pre-trained model,
with code from [training run 23](http://github.com/sagnibak/Turbulence#run-23).

**Turbulence_Model_1.h5:** This is the last pre-trained model,
with code from [training run 24](http://github.com/sagnibak/Turbulence#run-24).

**LICENCE:** I want to share this code freely for everyone to 
look at and use. All I want is credit, in case you modify and
redistribute the code. Neither am I asking for any money,
nor am I claiming any liability in case something goes wrong
on your end; I am more than happy to help, but I am not
getting sued. I have made sure that all the code in the 
`master` branch works exactly as it should.

**.gitignore:** This file contains the names of those files
that will not be uploaded to the repository. For learning
putposes, this is not important.



## The Training Runs (Hyperparameter Optimization)
Here, I have documented all the changes that I have made to the hyperparmeters. In each case, I have included only those snippets of code that I have changed, and included the accuracy and loss for each one after training.
#### Run 1
```python
epochs = 1000, batch_size = 100,
'adam', 'categorical_crossentropy',

model.add(Dense(20, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.712320870909  
Test accuracy: 0.666692414645*

#### Run 2
```python
epochs = 5000
```

*Test loss: 0.600918423797   
Test accuracy: 0.714969874865*

#### Run 3
```python 
model.add(Dense(30, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.586811004088  
Test accuracy: 0.727792368299 (actually maxed around 0.7347)*

#### Run 4
```python
model.add(Dense(30, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(20, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: {aborted after remaining stuck for 500 epochs}  
Test accuracy: {aborted after staying stuck for 500 epochs}*

#### Run 5
```python
model.add(Dense(32, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(8, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.603847062507  
Test accuracy: 0.705314382821 (actually maxed around 0.7225)*

#### Run 6
```python
model.add(Dense(32, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.599734134229  
Test accuracy: 0.71612853391 (actually maxed around 0.7237)*

#### Run 7
```python
model.add(Dense(28, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.5797  
Test accuracy: 0.7404*

#### Run 8
```python
model.add(Dense(28, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: {not recorded}  
Test accuracy: 0.69<accuracy<0.70*

#### Run 9
```python
model.add(Dense(28, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: {aborted after staying stuck for 800 epochs}  
Test accuracy: {aborted after staying stuck for 800 epochs}*

#### Run 10
```python
model.add(Dense(28, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: {aborted after staying stuck for 1200 epochs}  
Test accuracy: {aborted after staying stuck for 1200 epochs}*

#### Run 11
```python
model.add(Dense(28, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: {aborted after staying stuck for 500 epochs}  
Test accuracy: {aborted after staying stuck for 500 epochs}*

#### Run 12
```python
model = Sequential()
model.add(Dense(28, activation='sigmoid', input_dim=100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softplus'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.615324148614  
Test accuracy: 0.707399969102  
Calm Test loss: 6.19425047046  
Calm Test accuracy: 0.215362997658*   

*OVERFITTING!!!*

#### Run 13
```python
model.add(Dense(28, activation='sigmoid', input_dim=100))
model.add(Dropout(0.4))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='softplus'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```
---
*This model learnt nothing in 5000 epochs due to the enormous amounts of dropout.*

*I think that the reason that the model is not training properly is that the input layer
is only 100 units wide, scanning over a little than 0.5 seconds of data at once. So, I
am now modifying the data to allow the MLP to have 500 neurons in the input layer. This
will allow the network to scan 2.5 seconds into the past, which should increase accuracy.*

*As of now, it looks like there is something wrong with the 500-neuron architecture. I
tried different 2- and 3-layer configurations with ReLU activation, all of which failed
to converge in 200-500 epochs. Then I tried a sigmoid architecture as well (the one that
worked the best with 100 input neurons), which also failed to converge. Now I want to
play around a little with the optimizer.*


--------------------------------------------------------------------------------------------------------------------
#### Here's all the code for Run 14 onwards 

```python
epochs = 25000  
batch_size = 100

print("")
print("VERSION TanhLM.1")

print("Loading data...")
dataset = np.loadtxt("Turbulence_Training.csv", delimiter=",")  # load the dataset
# testset = np.loadtxt("Turbulence_Testing_Calm.csv", delimiter=",")  # testset is a palindrome!!!

print("Processing the data...")
inputs = dataset[:, 0:1]
new_outputs = dataset[499:, 1:4]
# validation_outputs = testset[99:-1, 1:4]

# new_inputs = np.array([[]])
# for i in range(499, 37045):
#     new_inputs = np.append(new_inputs, np.array([inputs[i - 499:i + 1]]))
# new_inputs = np.reshape(new_inputs, (36546, 500))
# np.savetxt("Processed_Turbulence_Inputs_Big.csv", new_inputs, delimiter=',')

new_inputs = np.loadtxt("Processed_Turbulence_Inputs_Big.csv", delimiter=',')
# validation_inputs = np.loadtxt("Processed_Turbulence_Testing_Calm.csv", delimiter=',')

X_train = new_inputs[5000:29000, :]
X_test = new_inputs[24000:, :]
Y_train = new_outputs[5000:29000, :]
Y_test = new_outputs[24000:, :]

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

print("Building the model...")
model = Sequential()
model.add(Dense(28, activation='tanh', input_dim=500))
model.add(Dropout(0.2))
model.add(Dense(10, activation='tanh'))
model.add(Dropout(0.2))
# model.add(Dense(5, activation='softplus'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.summary()

sgd = SGD(lr=1000.0, decay=0.001, momentum=1.0e-1, nesterov=False)
model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
```

*Test loss: 0.43035825716  
Test accuracy: 0.790876241158 (max around 0.7937)*

#### Run 15
```python
model.add(Dense(32, activation='tanh', input_dim=500))
model.add(Dropout(0.2))
model.add(Dense(11, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.440905945767  
Test accuracy: 0.783463519212 (max around 0.7850)*

#### Run 16
```python
model.add(Dense(28, activation='tanh', input_dim=500))
model.add(Dropout(0.2))
model.add(Dense(10, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
```

*Test loss: 0.344701589349  
Test accuracy: 0.842101072326*

#### Run 17
```python
model.add(Dense(24, activation='tanh', input_dim=500))
model.add(Dropout(0.2)) 
model.add(Dense(10, activation='tanh'))
model.add(Dropout(0.2))  
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.353813521501  
Test accuracy: 0.837770340185*  

---

*Something to be noted about this training run is that the Test Accuracy was actually lower
than the accuracy on the training data. Possibly, this is a sign of overfitting. So I am
increasing both the dropouts from 0.2 to 0.25. Let's see what happens.*  

*The accuracy on the testing data maxed around 0.8370. One thing
to be noted is that it ket bouncing in the 0.80-0.83 ballpark for several thousand epochs.*  

---
#### Run 18
```python
model.add(Dense(20, activation='tanh', input_dim=500))  
model.add(Dropout(0.2))
model.add(Dense(10, activation='tanh'))  
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

*Test loss: 0.416595364207    
Test accuracy: 0.800414479066 (max around 0.8270)*     

---
*This time too, the accuracy kept bouncing around in the 0.80-0.83 ballpark for the last 5000 epochs or so.*  

---

#### Run 19
```python
model = Sequential()
model.add(Dense(28, activation='tanh', input_dim=500))
model.add(Dropout(0.2))
model.add(Dense(8, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

initializers.TruncatedNormal(mean=1.0, stddev=0.05, seed=None)
```

*Test loss: 0.352511857044  
Test accuracy: 0.83877996005 (max around 0.8404)*

#### Run 20
```python
batch_size = 80
```

*Test loss: 0.35624565698  
Test accuracy: 0.838036028407 (max around 0.8442)*

#### Run 21
```python
batch_size = 70
```

*Test loss: 0.354777779751  
Test accuracy: 0.839205062011 (max around 0.8457)*



***THE BEGINNING OF THE END***
----------------------------------------------------------------------------------------------
#### Run 22
```python
batch_size = 60
```

*I also decided to finally process the Turbulence_Testing_Calm dataset for the
500-input-neuron MLP, so now there is FINALLY some validation data that the model
has really NEVER seen before. If the model, which is consistently classifying 5
out of 6 examples correctly, performs well on this dataset (unlike the 100-input-neuron
MLP that I had designed earlier), then I will finally put this code on GitHub and
call it a day.* * *fingers crossed* *  


*Test loss: 0.3521712085    
Test accuracy: 0.837345238072    
Calm Test loss: 0.0760556243008    
Calm Test accuracy: 1.0*  

---

*I CAN'T BELIEVE THAT THIS IS REAL!!! Okay. So the neural network does know the first thing
about turbulence. But before I upload this code, I want to make sure that it can correctly
identify turbulent motion as well. Then I will throw in the towel.*  

---

*Turbulent Test loss: 0.408611278705  
Turbulent Test accuracy: 0.80676939629*  
   
---

*I am impressed with the performance of this MLP. Even though it did not classify turbulent
weather as accurately as it classified calm weather, it performed pretty well, proving that
it is generalizing. It's also worth noting that turbulent data is a little hard to differentiate
from moderately turbulent weather, even for a human, so this MLP performed pretty well.*  

---------------------------------------------------------------------------------------------------
#### Run 23
```python
batch_size = 50
```

*Test loss: 0.354188750672  
Test accuracy: 0.839630161616 (max around 0.8429)  
Calm Test loss: 0.0697081955129  
Calm Test accuracy: 1.0  
Turbulent Test loss: 0.42392697862  
Turbulent Test accuracy: 0.800308980824*  

#### Run 24
```python
model.add(Dense(28, activation='tanh', input_dim=500))  
model.add(Dropout(0.2))
model.add(Dense(10, activation='tanh'))  
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

*Test loss: 0.386683792073  
Test accuracy: 0.82841809165 (max around 0.8455)    
Calm Test loss: 0.046252482396  
Calm Test accuracy: 1.0  
Turbulent Test loss: 0.459521576364  
Turbulent Test accuracy: 0.788030181484*  
