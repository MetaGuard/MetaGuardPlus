import time
import numpy as np
from common.miller import millerfeatures
from common.metaguard import metaguard
from common.metaguardplus import metaguardplus

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
replays = np.load('../data/user-identification/user-replays.npy')

# Control group
print("Processing control feautres...")
miller = millerfeatures(replays)
np.save('./anonymity/miller-control', miller)

# MetaGuard group
print("Processing MetaGuard feautres...")
anonymized = metaguard(replays.reshape(-1, 900, 21))
miller = millerfeatures(np.array(anonymized).reshape(-1, 20, 900, 21))
np.save('./anonymity/miller-metaguard', miller)

# MetaGuard++ group
print("Processing MetaGuard++ feautres...")
noise = np.random.normal(size=(10000,32)).astype('float16')
noise = np.repeat(noise, 10, axis=0)
anonymized = metaguardplus(replays.reshape(-1, 900, 21), noise)
miller = millerfeatures(np.array(anonymized).reshape(-1, 20, 900, 21))
np.save('./anonymity/miller-metaguardplus', miller)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
