import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
controlFeatures = np.load('./anonymity/miller-control.npy')
controlTrainX = controlFeatures[:,:300].reshape(-1, 105)
controlTestX = controlFeatures[:,300:].reshape(-1, 105)

metaguardFeatures = np.load('./anonymity/miller-metaguard.npy')
metaguardTrainX = metaguardFeatures[:,:300].reshape(-1, 105)
metaguardTestX = metaguardFeatures[:,300:].reshape(-1, 105)

metaguardPlusFeatures = np.load('./anonymity/miller-metaguardplus.npy')
metaguardPlusTrainX = metaguardPlusFeatures[:,:300].reshape(-1, 105)
metaguardPlusTestX = metaguardPlusFeatures[:,300:].reshape(-1, 105)

N = controlFeatures.shape[0]
trainY = np.repeat(np.arange(N), 300)
testY = np.repeat(np.arange(N), 300)

# Train a random forest model
def trainrf(trainX, trainY):
    rf = RandomForestClassifier(
        n_estimators=32,
        n_jobs=32,
        verbose=100
    )
    rf.fit(trainX, trainY)
    rf.verbose = 0
    return rf

# Test a random forest model
def testrf(rf, testX, name):
    print("--", name, "--")
    predY = rf.predict(testX)
    acc = accuracy_score(testY, predY)
    print("Accuracy (Per Sample): " + str(acc))
    predY = [np.bincount(x).argmax() for x in predY.reshape(-1, 300)]
    acc = accuracy_score(list(range(N)), predY)
    print("Accuracy (Per User): " + str(acc))

# Train non-adaptive miller model
print("Training Miller model...")
rf = trainrf(controlTrainX, trainY)
testrf(rf, controlTestX, "Control")
testrf(rf, metaguardTestX, "MetaGuard")
testrf(rf, metaguardPlusTestX, "MetaGuard++")

# Train adaptive miller model (MetaGuard)
print("Training Miller model (MetaGuard)...")
rf = trainrf(metaguardTrainX, trainY)
testrf(rf, controlTestX, "Control")
testrf(rf, metaguardTestX, "MetaGuard")
testrf(rf, metaguardPlusTestX, "MetaGuard++")

# Train adaptive miller model (MetaGuard++)
print("Training Miller model (MetaGuard++)...")
rf = trainrf(metaguardPlusTrainX, trainY)
testrf(rf, controlTestX, "Control")
testrf(rf, metaguardTestX, "MetaGuard")
testrf(rf, metaguardPlusTestX, "MetaGuard++")

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
