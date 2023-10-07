# Import libraries
import time
import gc
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K

# Configuration options
NUM_USERS =  500                        # Number of users to include
TRAIN_SIZE = 400                        # Replays per user for training
VAL_SIZE =   50                         # Replays per user for validation
TEST_SIZE =  50                         # Replays per user for testing

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
data = np.load('./data/user-replays.npy')
trainX = data[:,:TRAIN_SIZE].reshape(-1, 900, 21)
valX = data[:,TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE].reshape(-1, 900, 21)
testX = data[:,TRAIN_SIZE+VAL_SIZE:].reshape(-1, 900, 21)
trainY = np.repeat(np.arange(NUM_USERS), TRAIN_SIZE)
valY = np.repeat(np.arange(NUM_USERS), VAL_SIZE)
testY = np.repeat(np.arange(NUM_USERS), TEST_SIZE)

# Convert the labels to one-hot vectors
trainY = tf.one_hot(trainY, depth=NUM_USERS)
valY = tf.one_hot(valY, depth=NUM_USERS)
testY = tf.one_hot(testY, depth=NUM_USERS)

# Train and test an LSTM Funnel model with Keras
def run(trainX, trainY, testX, testY, valX, valY):
    print("Train Shape: ", trainX.shape)

    # Define the identification model architecture
    model = keras.Sequential([
        layers.LSTM(256, input_shape=(900, trainX.shape[-1]), return_sequences=True),
        layers.AveragePooling1D(pool_size=30),
        layers.LSTM(256),
        layers.Dense(256),
        layers.Dense(256),
        layers.Dense(NUM_USERS, activation="softmax")
    ], name='user-identification')

    # Compile the model with loss and metrics
    model.compile(
        loss="categorical_crossentropy", # Use categorical crossentropy as the loss function
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
    model.fit(trainX, trainY,
        epochs=500,
        batch_size=256,
        shuffle=True,
        validation_data=(valX, valY),
        callbacks=[early_stopping]
    )

    # Clear GPU memory
    K.clear_session()
    gc.collect()

    # Test per-sample accuracy
    predY = model.predict(testX)
    realY = np.repeat(np.arange(NUM_USERS), TEST_SIZE)
    acc = accuracy_score(realY, predY.argmax(axis=1))
    print("Accuracy (Per Sample): " + str(acc))

    # Test per-user accuracy
    with np.errstate(divide='ignore'):
        predY = np.log(predY).reshape(-1, TEST_SIZE, NUM_USERS).sum(axis=1).argmax(axis=1)
    acc = accuracy_score(list(range(NUM_USERS)), predY)
    print("Accuracy (Per User): " + str(acc))

# Train with all features
print("Running with all features...")
run(trainX, trainY, testX, testY, valX, valY)

print("Running with only hands...")
run(trainX[:,:,7:], trainY, testX[:,:,7:], testY, valX[:,:,7:], valY)

print("Running with only hand rotations...")
ax = [10, 11, 12, 13, 17, 18, 19, 20]
run(trainX[:,:,ax], trainY, testX[:,:,ax], testY, valX[:,:,ax], valY)

print("Running with only left hand rotation...")
ax = [10, 11, 12, 13]
run(trainX[:,:,ax], trainY, testX[:,:,ax], testY, valX[:,:,ax], valY)

print("Running with only left hand rotation magnitude...")
run(trainX[:,:,[13]], trainY, testX[:,:,[13]], testY, valX[:,:,[13]], valY)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
