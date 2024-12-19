# Slide page: 60
import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.Optimizers import SGD

N, D, H = 64, 1000, 100 

# Define model object as a sequence of layers
model = Sequential()
model.add(Dense(input_dim = D, output_dim = H))
model.add(Activation('relu'))
model.add(Dense(input_dim = H, output_dim = D))

# Define optimizer object
optimizer = SGD(lr = 1e0)

# Build the model, specify loss function
model.compile(loss= 'mean_squared_error', optimizer = optimizer)

x = np.random.randn(N, D)
y = np.random.randn(N, D)

# Train the model with a single line!
history = model.fit(x, y, nb_epoch = 50, batch_size = N, verbose = 0)