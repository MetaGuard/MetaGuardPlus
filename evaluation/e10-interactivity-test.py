import os
import time
import numpy as np
from tqdm import tqdm
from common.main import calculate_score_assuming_valid_times
from common.main.interpretMapFiles import create_map
from bsor.Bsor import make_bsor

# Start measuring performance
start_time = time.time()

# Get list of replays in directory
replays = list(os.listdir("./interactivity/control"))

# Test all replays
score_diff = []
preswing_diff = []
postswing_diff = []
accuracy_diff = []
for replay in tqdm(replays):
    try:
        # Load map file
        map = replay.split('-')[-1].split('.')[0]
        mapdata = create_map("./interactivity/maps/" + map)

        # Load replay files
        with open("./interactivity/control/" + replay, 'rb') as f:
            control = make_bsor(f)
        with open("./interactivity/metaguardplus/" + replay, 'rb') as f:
            metaguardplus = make_bsor(f)

        # Simulate results
        c_score, c_evs = calculate_score_assuming_valid_times(mapdata, control)
        mgp_score, mgp_evs = calculate_score_assuming_valid_times(mapdata, metaguardplus)
        if (c_score == 0): continue

        # Log score differences
        score_diff.append(abs(mgp_score - c_score) / c_score)
        for n in range(30):
            c_parts = c_evs[n].get_score_breakdown()
            mgp_parts = mgp_evs[n].get_score_breakdown()
            if (c_parts[0] != 0): preswing_diff.append(abs(mgp_parts[0] - c_parts[0]) / c_parts[0])
            if (c_parts[1] != 0): postswing_diff.append(abs(mgp_parts[1] - c_parts[1]) / c_parts[1])
            if (c_parts[2] != 0): accuracy_diff.append(abs(mgp_parts[2] - c_parts[2]) / c_parts[2])

    except Exception as e:
        continue

print("Score Difference", np.mean(score_diff))
print("Preswing Difference", np.mean(np.abs(preswing_diff)))
print("Postswing Difference", np.mean(np.abs(postswing_diff)))
print("Accuracy Difference", np.mean(np.abs(accuracy_diff)))

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
