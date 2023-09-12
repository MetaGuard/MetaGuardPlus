import time
import numpy as np
from common.metaguard import metaguard
from common.metaguardplus import metaguardplus

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
replays = np.load('../data/user-identification/user-replays.npy')

# Control group
print("Processing control feautres...")
np.save('./anonymity/funnel-control', replays)

# MetaGuard group
print("Processing MetaGuard feautres...")
anonymized = metaguard(replays.reshape(-1, 900, 21))
np.save('./anonymity/funnel-metaguard', np.array(anonymized).reshape(-1, 20, 900, 21))

# MetaGuard++ group
print("Processing MetaGuard++ feautres...")
noise = np.random.normal(size=(2000,32)).astype('float16')
noise = np.repeat(noise, 10, axis=0)
anonymized = metaguardplus(replays.reshape(-1, 900, 21), noise)
np.save('./anonymity/funnel-metaguardplus', np.array(anonymized).reshape(-1, 20, 900, 21))

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
