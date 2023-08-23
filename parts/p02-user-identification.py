# Import libraries
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration options
NUM_USERS = 5000                        # Number of users for identification
TRAIN_SIZE = 10                         # Number of samples for each class for training
TEST_SIZE = 10                          # Number of samples for each class for testing
LAYER_SIZE = 256                        # Hidden size for DNN/LSTM layers
BATCH_SIZE = 256                        # Batch size for training and testing

# Start measuring performance
print("Loading input data...")
start_time = time.time()

# Load the input data
data = np.load("../data/user-identification/user-replays.npy")

# Split the data into train and test sets
train_data = data[:, :TRAIN_SIZE]
test_data = data[:, TRAIN_SIZE:]

# Reshape the data to fit the model input
train_data = train_data.reshape(-1, 900, 21)
test_data = test_data.reshape(-1, 900, 21)

# Create the labels for the data
train_labels = np.repeat(np.arange(NUM_USERS), TRAIN_SIZE)
test_labels = np.repeat(np.arange(NUM_USERS), TEST_SIZE)

# Convert the labels to one-hot vectors
train_labels = tf.one_hot(train_labels, depth=NUM_USERS)
test_labels = tf.one_hot(test_labels, depth=NUM_USERS)

# Free unused memory
del data

# Define the identification model architecture
model = keras.Sequential([
    layers.LSTM(LAYER_SIZE, input_shape=(900, 21), return_sequences=True),
    layers.AveragePooling1D(pool_size=30),
    layers.LSTM(LAYER_SIZE),
    layers.Dense(LAYER_SIZE),
    layers.Dense(LAYER_SIZE),
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
print("Training user identification model...")
model.fit(train_data, train_labels,
          epochs=500,
          batch_size=BATCH_SIZE,
          shuffle=True,
          validation_data=(test_data, test_labels),
          callbacks=[early_stopping])

# Save model
print("Saving user identification model...")
model.save('./models/user-identification.keras')

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
