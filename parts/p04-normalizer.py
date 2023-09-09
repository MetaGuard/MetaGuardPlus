# Import libraries
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration options
TRAIN_SIZE = 32768                      # Number of replays for training
VAL_SIZE = 4096                         # Number of replays for validation
TEST_SIZE = 4096                        # Number of replays for testing
LAYER_SIZE = 256                        # Hidden size for DNN/LSTM layers
BATCH_SIZE = 512                        # Batch size for training and testing

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
data = np.load("../data/action-similarity/train-true.npy")
data = np.concatenate([data[:,0], data[:,1]])
np.random.shuffle(data)

# Load the anonymizer model
anonymizer = keras.models.load_model('./models/anonymizer.keras')

# Anonymize the input data
print("Anonymizing input data...")
noise = np.random.normal(size=(len(data), 32)).astype('float16')
noisy = anonymizer.predict([data, noise], batch_size=BATCH_SIZE)

# Recenter the input data
print("Normalizing input data...")
data_mean = np.mean(data, axis=1, dtype=np.float64)
data_std = np.std(data, axis=1, dtype=np.float64)
data_mean = np.expand_dims(data_mean, 1).repeat(900,axis=1)
data_std = np.expand_dims(data_std, 1).repeat(900,axis=1)
anon_mean = np.mean(noisy, axis=1, dtype=np.float64)
anon_std = np.std(noisy, axis=1, dtype=np.float64)
anon_mean = np.expand_dims(anon_mean, 1).repeat(900,axis=1)
anon_std = np.expand_dims(anon_std, 1).repeat(900,axis=1)
noisy = (noisy - anon_mean) / anon_std
noisy = (noisy * data_std) + data_mean

# Split into train, test, validate sets
trainX = noisy[0:TRAIN_SIZE]
trainY = data[0:TRAIN_SIZE]
valX = noisy[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
valY = data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
testX = noisy[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE]
testY = data[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE]

# Define the normalizer model
normalizer = keras.Sequential([
    layers.LSTM(LAYER_SIZE, return_sequences=True),
    layers.TimeDistributed(layers.Dense(21))
], name='normalizer')

# Configure early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    mode="min",
    restore_best_weights=True
)

# Compile the model with loss and metrics
adam = keras.optimizers.Adam(learning_rate=0.001)
normalizer.compile(loss="mse", optimizer=adam, metrics=["mae"])

# Train the model on the train data
print("Training normalizer model...")
normalizer.fit(trainX, trainY, epochs=500, batch_size=BATCH_SIZE, shuffle=True, validation_data=(valX, valY), callbacks=[early_stopping])

# Recompile the model with lower learning rate
adam = keras.optimizers.Adam(learning_rate=0.0001)
normalizer.compile(loss="mse", optimizer=adam, metrics=["mae"])

# Train the model on the train data
normalizer.fit(trainX, trainY, epochs=500, batch_size=BATCH_SIZE, shuffle=True, validation_data=(valX, valY), callbacks=[early_stopping])

# Recompile the model with lower learning rate
adam = keras.optimizers.Adam(learning_rate=0.00001)
normalizer.compile(loss="mse", optimizer=adam, metrics=["mae"])

# Train the model on the train data
normalizer.fit(trainX, trainY, epochs=500, batch_size=BATCH_SIZE, shuffle=True, validation_data=(valX, valY), callbacks=[early_stopping])

# Test the anonymizer
print("Testing normalizer model...")
normalizer.evaluate(testX, testY, batch_size=BATCH_SIZE)

# Save model
print("Saving normalizer model...")
normalizer.save('./models/normalizer.keras')

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
