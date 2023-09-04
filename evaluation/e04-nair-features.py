import time
import numpy as np
from common.nair import nairfeatures
from common.metaguard import metaguard
from common.metaguardplus import metaguardplus

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
replays = np.load('../data/user-identification/user-replays.npy')
notes = np.load('../data/user-identification/user-notes.npy')

# Control group
print("Processing control feautres...")
nair = nairfeatures(replays, notes)
np.save('./anonymity/nair-control', nair)

# MetaGuard group
print("Processing MetaGuard feautres...")
anonymized = metaguard(replays.reshape(-1, 900, 21))
nair = nairfeatures(np.array(anonymized).reshape(-1, 20, 900, 21), notes)
np.save('./anonymity/nair-metaguard', nair)

# MetaGuard++ group
print("Processing MetaGuard++ feautres...")
noise = np.random.normal(size=(1000,32)).astype('float16')
noise = np.repeat(noise, 10, axis=0)
anonymized = metaguardplus(replays.reshape(-1, 900, 21), noise)
nair = nairfeatures(np.array(anonymized).reshape(-1, 20, 900, 21), notes)
np.save('./anonymity/nair-metaguardplus', nair)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
