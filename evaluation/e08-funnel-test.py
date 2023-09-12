import time
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
controlFeatures = np.load('./anonymity/funnel-control.npy')
controlTrainX = controlFeatures[:,:10].reshape(-1, 900, 21)
controlTestX = controlFeatures[:,10:].reshape(-1, 900, 21)

metaguardFeatures = np.load('./anonymity/funnel-metaguard.npy')
metaguardTrainX = metaguardFeatures[:,:10].reshape(-1, 900, 21)
metaguardTestX = metaguardFeatures[:,10:].reshape(-1, 900, 21)

metaguardPlusFeatures = np.load('./anonymity/funnel-metaguardplus.npy')
metaguardPlusTrainX = metaguardPlusFeatures[:,:10].reshape(-1, 900, 21)
metaguardPlusTestX = metaguardPlusFeatures[:,10:].reshape(-1, 900, 21)

N = controlFeatures.shape[0]
trainY = np.repeat(np.arange(N), 10)
testY = np.repeat(np.arange(N), 10)

# Convert the labels to one-hot vectors
train_labels = tf.one_hot(trainY, depth=N)
test_labels = tf.one_hot(testY, depth=N)

# Train a Keras LSTM model
def train(trainX, trainY):
    # Define the identification model architecture
    model = keras.Sequential([
        layers.LSTM(256, input_shape=(900, 21), return_sequences=True),
        layers.AveragePooling1D(pool_size=30),
        layers.LSTM(256),
        layers.Dense(256),
        layers.Dense(256),
        layers.Dense(N, activation="softmax")
    ], name='user-identification')

    # Compile the model with loss and metrics
    model.compile(
        loss="categorical_crossentropy", # Use categorical crossentropy as the loss function
        optimizer="adam", # Use adam as the optimizer
        metrics=["accuracy"] # Use accuracy as the metric
    )

    # Train the model on the train data
    model.fit(trainX, trainY, epochs=50, batch_size=256, shuffle=True)

    return model

# Test a Keras LSTM model
def test(clf, testX, name):
    print("--", name, "--")
    predY = clf.predict(testX)
    acc = accuracy_score(testY, predY.argmax(axis=1))
    print("Accuracy (Per Sample): " + str(acc))

    with np.errstate(divide='ignore'):
        predY = np.log(predY).reshape(-1, 10, N).sum(axis=1).argmax(axis=1)
    acc = accuracy_score(list(range(N)), predY)
    print("Accuracy (Per User): " + str(acc))

# Train non-adaptive funnel model
print("Training LSTM model...")
model = train(controlTrainX, train_labels)
test(model, controlTestX, "Control")
test(model, metaguardTestX, "MetaGuard")
test(model, metaguardPlusTestX, "MetaGuard++")

# Train adaptive funnel model (MetaGuard)
print("Training LSTM model (MetaGuard)...")
model = train(metaguardTrainX, train_labels)
test(model, controlTestX, "Control")
test(model, metaguardTestX, "MetaGuard")
test(model, metaguardPlusTestX, "MetaGuard++")

# Train adaptive funnel model (MetaGuard++)
print("Training LSTM model (MetaGuard++)...")
model = train(metaguardPlusTrainX, train_labels)
test(model, controlTestX, "Control")
test(model, metaguardTestX, "MetaGuard")
test(model, metaguardPlusTestX, "MetaGuard++")

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
