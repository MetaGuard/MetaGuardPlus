# Import libraries
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K

# Configuration options
TRAIN_SIZE = 25000                      # Number of pairs for each class for training
TEST_SIZE = 5000                        # Number of pairs for each class for testing
LAYER_SIZE = 256                        # Hidden size for DNN/LSTM layers
BATCH_SIZE = 256                        # Batch size for training and testing

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
train_true = np.load("../data/user-similarity/train-true.npy")[:TRAIN_SIZE]
train_false = np.load("../data/user-similarity/train-false.npy")[:TRAIN_SIZE]
trainX1 = np.concatenate([train_true[:,0], train_false[:,0]]).astype('float16')
trainX2 = np.concatenate([train_true[:,1], train_false[:,0]]).astype('float16')
test_true = np.load("../data/user-similarity/test-true.npy")
test_false = np.load("../data/user-similarity/test-false.npy")
testX1 = np.concatenate([test_true[:,0], test_false[:,0]]).astype('float16')
testX2 = np.concatenate([test_true[:,1], test_false[:,0]]).astype('float16')

# Compute random noise
def getNoise(size):
    noise1 = np.random.normal(size=(size, 32)).astype('float16')
    noise2 = np.random.normal(size=(size, 32)).astype('float16')
    noise3 = np.random.normal(size=(size, 32)).astype('float16')
    N1 = np.concatenate([noise1, noise2]).astype('float16')
    N2 = np.concatenate([noise1, noise3]).astype('float16')
    return N1, N2
trainN1, trainN2 = getNoise(TRAIN_SIZE)
testN1, testN2 = getNoise(TEST_SIZE)

# Set training targets
trainY = np.concatenate([np.ones(TRAIN_SIZE), np.zeros(TRAIN_SIZE)])
testY = np.concatenate([np.ones(TEST_SIZE), np.zeros(TEST_SIZE)])

# Load pre-trained models
action_similarity = keras.models.load_model('./models/action-similarity.keras')
action_similarity.trainable = False
action_encoder = action_similarity.layers[2]
action_encoder.trainable = False
user_similarity = keras.models.load_model('./models/user-similarity.keras')
user_similarity.trainable = False

# Compute action targets
trainA1 = action_encoder.predict(trainX1, batch_size=BATCH_SIZE).astype('float16')
trainA2 = action_encoder.predict(trainX2, batch_size=BATCH_SIZE).astype('float16')
testA1 = action_encoder.predict(testX1, batch_size=BATCH_SIZE).astype('float16')
testA2 = action_encoder.predict(testX2, batch_size=BATCH_SIZE).astype('float16')

# Define the anonymizer model
x = layers.Input(shape=(900, 21))
r = layers.Input(shape=32)
rv = layers.RepeatVector(900)(r)
l = layers.Conv1D(64, 30, padding="causal")(x)
xrl = layers.Concatenate(axis=-1)([x, rv, l])
o = layers.TimeDistributed(layers.Dense(128))(xrl)
o = layers.TimeDistributed(layers.Dense(64))(o)
o = layers.TimeDistributed(layers.Dense(21))(o)
anonymizer = keras.Model(inputs=[x, r], outputs=o, name='anonymizer')

# Compile the anonymizer model
anonymizer.compile(optimizer='adam', loss='mse')

# Pretrain the anonymizer
print("Pretraining anonymizer model...")
anonymizer.fit([trainX1, trainN1], trainX1, batch_size=BATCH_SIZE, epochs=20)

# Define the training model architecture
x1 = layers.Input(shape=(900, 21))
x2 = layers.Input(shape=(900, 21))
n1 = layers.Input(shape=32)
n2 = layers.Input(shape=32)
y1 = anonymizer([x1, n1])
y2 = anonymizer([x2, n2])
us = user_similarity([y1, y2])
as1 = action_similarity([x1, y1])
as2 = action_similarity([x2, y2])
ae1 = action_encoder(y1)
ae2 = action_encoder(y2)
model = keras.Model(inputs=[x1, x2, n1, n2], outputs=[us, as1, as2, ae1, ae2])

# Configure early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=5,
    verbose=1,
    mode="min",
    restore_best_weights=True
)

# Train model with learning rate
trainOnes = np.ones(TRAIN_SIZE*2).astype('float16')
def train(lr):
    # Compile the training model
    adam = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss={
            'user-similarity': 'mae',
            'action-similarity': 'mae',
            'action-similarity_1': 'mae',
            'action-encoder': 'mae',
            'action-encoder_1': 'mae'
        },
        loss_weights={
            'user-similarity': 1,
            'action-similarity': 1,
            'action-similarity_1': 1,
            'action-encoder': 0.1,
            'action-encoder_1': 0.1
        },
        metrics={
            'user-similarity': 'accuracy',
            'action-similarity': 'accuracy',
            'action-similarity_1': 'accuracy'
        },
        optimizer=adam
    )

    # Train the anonymizer
    model.fit([trainX1, trainX2, trainN1, trainN2],
                [trainY, trainOnes, trainOnes, trainA1, trainA2],
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping],
                shuffle=True,
                epochs=500)

# Train the anonymizer
print("Training anonymizer model...")
train(0.0001)
train(0.00001)
train(0.000001)

# Test the anonymizer
print("Testing anonymizer model...")
testOnes = np.ones(TEST_SIZE*2).astype('float16')
model.evaluate([testX1, testX2, testN1, testN2], [testY, testOnes, testOnes, testA1, testA2], batch_size=BATCH_SIZE)

# Save model
print("Saving anonymizer model...")
anonymizer.save('./models/anonymizer.keras')

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
