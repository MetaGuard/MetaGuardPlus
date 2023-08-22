# Import libraries
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keras.backend as K

# Configuration options
TRAIN_SIZE = 50000                      # Number of pairs for each class for training
VAL_SIZE = 5000                         # Number of pairs for each class for testing
TEST_SIZE = 5000                        # Number of pairs for each class for validation
LAYER_SIZE = 256                        # Hidden size for DNN/LSTM layers
BATCH_SIZE = 256                        # Batch size for training and testing

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
train_true = np.load("../data/action-similarity/train-true.npy")
train_false = np.load("../data/action-similarity/train-false.npy")
validate_true = np.load("../data/action-similarity/validate-true.npy")
validate_false = np.load("../data/action-similarity/validate-false.npy")
test_true = np.load("../data/action-similarity/test-true.npy")
test_false = np.load("../data/action-similarity/test-false.npy")

# Split the data into train and test sets
train = np.vstack([train_true, train_false])
validate = np.vstack([validate_true, validate_false])
test = np.vstack([test_true, test_false])

# Split the train and test inputs
trainX1 = train[:,0]
trainX2 = train[:,1]
valX1 = validate[:,0]
valX2 = validate[:,1]
testX1 = test[:,0]
testX2 = test[:,1]

# Create the labels for the data
trainY = np.hstack([np.ones(TRAIN_SIZE), np.zeros(TRAIN_SIZE)])
valY = np.hstack([np.ones(VAL_SIZE), np.zeros(VAL_SIZE)])
testY = np.hstack([np.ones(TEST_SIZE), np.zeros(TEST_SIZE)])

# Define the encoder model architecture
encoder = keras.Sequential([
    layers.LSTM(LAYER_SIZE, input_shape=(900, 21), return_sequences=True),
    layers.AveragePooling1D(pool_size=30),
    layers.LSTM(LAYER_SIZE),
    layers.Dense(LAYER_SIZE),
    layers.Dense(LAYER_SIZE)
], name='action-encoder')

# Define the action similarity model architecture
x1 = layers.Input(shape=(900, 21))
x2 = layers.Input(shape=(900, 21))
e1 = encoder(x1)
e2 = encoder(x2)
d = layers.Subtract()([e1, e2])
d = layers.Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)))(d)
o = layers.Dense(1, activation='sigmoid')(d)
model = keras.Model(inputs=[x1, x2], outputs=o, name='action-similarity')

# Compile the model with loss and metrics
model.compile(
    loss="binary_crossentropy", # Use categorical crossentropy as the loss function
    optimizer="adam", # Use adam as the optimizer
    metrics=["accuracy"] # Use accuracy as the metric
)

# Configure early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=25,
    verbose=1,
    mode="max",
    restore_best_weights=True
)

# Train the model on the train data
print("Training action similarity model...")
model.fit([trainX1, trainX2], trainY,
           epochs=500,
           batch_size=BATCH_SIZE,
           shuffle=True,
           validation_data=([valX1, valX2], valY),
           callbacks=[early_stopping])

# Evaluate the model on the test data
print("Testing action similarity model...")
model.evaluate([testX1, testX2], testY, batch_size=BATCH_SIZE)

# Save model
print("Saving action similarity model...")
model.save('./models/action-similarity.keras')
encoder.save('./models/action-encoder.keras')

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
