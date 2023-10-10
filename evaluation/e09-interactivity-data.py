import os
import time
import numpy as np
from tqdm import trange
from bsor.Bsor import make_bsor
from common.xror import XROR
from common.parse import handle_bsor
from common.metaguard import metaguard
from common.metaguardplus import metaguardplus
import shutil

# Configuration options
REPLAY_DIR = "Z:/beatleader/replays/"   # Directory with replays
MAP_DIR = "Z:/beatsaver/maps/"          # Directory with maps
NUM_REPLAYS = 1000                      # Number of replays to include

# Start measuring performance
start_time = time.time()

# Get list of replays
with open('../data/user-identification/all-replays.txt') as file:
    replays = file.read().splitlines()
np.random.shuffle(replays)

# Create output folders
for folder in ['maps', 'control', 'metaguardplus']:
    if not os.path.exists('./interactivity/' + folder):
        os.makedirs('./interactivity/' + folder)

# Process replays
i, j = 0, 0
for k in trange(NUM_REPLAYS):
    while (j <= k):
        replay = replays[i]
        i += 1
        try:
            # Parse replay
            with open(REPLAY_DIR + replay, 'rb') as f:
                xror = XROR.fromBSOR(f)
            frames = handle_bsor('Z:/beatleader/replays/' + replay)
            times = np.arange(0, 30, 1/30).reshape(900,1)

            # Copy map
            map = replay.split('-')[-1].split('.')[0]
            src = MAP_DIR + map + ".zip"
            if not os.path.exists(src): continue
            if os.path.getsize(src) <= 1024: continue
            dst = "./interactivity/maps/" + map + ".zip"
            shutil.copyfile(src, dst)

            # Control replays
            xror.data['frames'] = np.hstack([times, frames])
            bsor = xror.toBSOR()
            with open('./interactivity/control/' + replay, 'wb') as f:
                f.write(bsor)

            # MetaGuard++ replays
            noise = np.random.normal(size=32).astype('float16')
            frames = metaguardplus(np.repeat([frames], 500, axis=0), np.repeat([noise], 500, axis=0))[0]
            xror.data['frames'] = np.hstack([times, frames])
            bsor = xror.toBSOR()
            with open('./interactivity/metaguardplus/' + replay, 'wb') as f:
                f.write(bsor)

            j += 1
        except Exception as e:
            pass

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
