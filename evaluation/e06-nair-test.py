import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier, log_evaluation

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
controlFeatures = np.load('./anonymity/nair-control.npy')
controlTrainX = controlFeatures[:,:300].reshape(-1, 211)
controlTestX = controlFeatures[:,300:].reshape(-1, 211)

metaguardFeatures = np.load('./anonymity/nair-metaguard.npy')
metaguardTrainX = metaguardFeatures[:,:300].reshape(-1, 211)
metaguardTestX = metaguardFeatures[:,300:].reshape(-1, 211)

metaguardPlusFeatures = np.load('./anonymity/nair-metaguardplus.npy')
metaguardPlusTrainX = metaguardPlusFeatures[:,:300].reshape(-1, 211)
metaguardPlusTestX = metaguardPlusFeatures[:,300:].reshape(-1, 211)

N = controlFeatures.shape[0]
trainY = np.repeat(np.arange(N), 300)
testY = np.repeat(np.arange(N), 300)

# Train a LightGBM model
def train(trainX, trainY):
    clf = LGBMClassifier(boosting_type='goss', colsample_bytree=0.6933333333333332, learning_rate=0.1, \
    max_bin=63, max_depth=-1, min_child_weight=7, min_child_samples=20, \
    min_split_gain=0.9473684210526315, n_estimators=200, histogram_pool_size=-1, \
    num_leaves=33, reg_alpha=0.7894736842105263, reg_lambda=0.894736842105263, \
    subsample=1, n_jobs=16, objective='multiclass', device_type='gpu')

    clf.fit(trainX, trainY, eval_set=[(trainX, trainY)], eval_metric='multi_error', callbacks=[log_evaluation()])

    return clf

# Test a LightGBM model
def test(clf, testX, name):
    print("--", name, "--")
    predY = clf.predict_proba(testX)
    acc = accuracy_score(testY, predY.argmax(axis=1))
    print("Accuracy (Per Sample): " + str(acc))

    predY = np.log(predY).reshape(-1, 300, N).sum(axis=1).argmax(axis=1)
    acc = accuracy_score(list(range(N)), predY)
    print("Accuracy (Per User): " + str(acc))

# Train non-adaptive nair model
print("Training Nair model...")
model = train(controlTrainX, trainY)
test(model, controlTestX, "Control")
test(model, metaguardTestX, "MetaGuard")
test(model, metaguardPlusTestX, "MetaGuard++")

# Train adaptive nair model (MetaGuard)
print("Training Nair model (MetaGuard)...")
model = train(metaguardTrainX, trainY)
test(model, controlTestX, "Control")
test(model, metaguardTestX, "MetaGuard")
test(model, metaguardPlusTestX, "MetaGuard++")

# Train adaptive nair model (MetaGuard++)
print("Training Nair model (MetaGuard++)...")
model = train(metaguardPlusTrainX, trainY)
test(model, controlTestX, "Control")
test(model, metaguardTestX, "MetaGuard")
test(model, metaguardPlusTestX, "MetaGuard++")

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
