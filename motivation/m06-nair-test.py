# Import libraries
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier, log_evaluation

# Configuration options
NUM_USERS =  500                        # Number of users to include
TRAIN_SIZE = 8000                       # Features per user for training
TEST_SIZE =  1000                       # Features per user for testing

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
data = np.load('./data/nair.npy')
trainX = data[:,:TRAIN_SIZE].reshape(-1, 211)
testX = data[:,TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE].reshape(-1, 211)
trainY = np.repeat(np.arange(NUM_USERS), TRAIN_SIZE)
testY = np.repeat(np.arange(NUM_USERS), TEST_SIZE)

# Train a LightGBM model
def train(trainX, trainY):
    clf = LGBMClassifier(boosting_type='goss', colsample_bytree=0.6933333333333332, learning_rate=0.1, \
    max_bin=63, max_depth=-1, min_child_weight=7, min_child_samples=20, \
    min_split_gain=0.9473684210526315, n_estimators=25, histogram_pool_size=-1, \
    num_leaves=33, reg_alpha=0.7894736842105263, reg_lambda=0.894736842105263, \
    subsample=1, n_jobs=16, objective='multiclass', device_type='gpu')

    clf.fit(trainX, trainY, eval_set=[(trainX, trainY)], eval_metric='multi_error', callbacks=[log_evaluation()])

    return clf

# Test a LightGBM model
def test(clf, testX):
    predY = clf.predict_proba(testX)
    acc = accuracy_score(testY, predY.argmax(axis=1))
    print("Accuracy (Per Sample): " + str(acc))

    predY = np.log(predY).reshape(-1, TEST_SIZE, NUM_USERS).sum(axis=1).argmax(axis=1)
    acc = accuracy_score(list(range(NUM_USERS)), predY)
    print("Accuracy (Per User): " + str(acc))

# Train nair model
print("Training model...")
model = train(trainX, trainY)
test(model, testX)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
