# Import libraries
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import trange

# Configuration options
NUM_USERS =  500                        # Number of users to include
TRAIN_SIZE = 12000                      # Features per user for training
TEST_SIZE =  1500                       # Features per user for testing

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
data = np.load('./data/miller.npy')
trainX = data[:,:TRAIN_SIZE].reshape(-1, 105)
testX = data[:,TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE].reshape(-1, 105)
trainY = np.repeat(np.arange(NUM_USERS), TRAIN_SIZE)
testY = np.repeat(np.arange(NUM_USERS), TEST_SIZE)

# Train a random forest model
def trainrf(trainX, trainY):
    rf = RandomForestClassifier(
        n_estimators=16,
        n_jobs=16,
        max_depth=16,
        verbose=100
    )
    rf.fit(trainX, trainY)
    rf.verbose = 0
    return rf

# Predict in batches
def batch_predict(rf, testX):
    BATCH_SIZE = 10000
    output = []
    for i in trange(testX.shape[0] // BATCH_SIZE):
        output.append(rf.predict(testX[BATCH_SIZE*i:BATCH_SIZE*(i+1)]))
    return np.hstack(output)

# Test a random forest model
def testrf(rf, testX):
    predY = batch_predict(rf, testX)
    acc = accuracy_score(testY, predY)
    print("Accuracy (Per Sample): " + str(acc))
    predY = [np.bincount(x).argmax() for x in predY.reshape(-1, TEST_SIZE)]
    acc = accuracy_score(list(range(NUM_USERS)), predY)
    print("Accuracy (Per User): " + str(acc))

# Train miller model
print("Training model...")
rf = trainrf(trainX, trainY)
testrf(rf, testX)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
