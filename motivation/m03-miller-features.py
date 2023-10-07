# Import libraries
import time
import numpy as np
import sys
sys.path.append("../evaluation")
from common.miller import millerfeatures

# Start measuring performance
start_time = time.time()

# Load the input data
print("Loading input data...")
replays = np.load('./data/user-replays.npy')

# Process feautres
print("Processing feautres...")
miller = millerfeatures(replays)
np.save('./data/miller', miller)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
