import os
import time
import numpy as np
from tqdm import tqdm, trange
from bsor.Bsor import make_bsor

# Configuration options
REPLAY_DIR = "Z:/beatleader/replays/"   # Directory with replays
TRAIN_SIZE = 50000                      # Number of pairs for each class for training
VAL_SIZE = 5000                         # Number of pairs for each class for testing
TEST_SIZE = 5000                        # Number of pairs for each class for validation
MIN_REPLAYS = 500                       # Minimum number of replays per user for inclusion

# Start measuring performance
start_time = time.time()

# Get list of replays in directory
print("Scanning replay directory...")
replays = list(os.listdir(REPLAY_DIR))
print("Found " + str(len(replays)) + " replays.")

# Group replays by users
print("Grouping replays...")
users = {}
user_names = []

for name in tqdm(replays):
    user = name.split('-')[0]
    if (len(user) > 17): continue

    if not user in users:
        users[user] = []
        user_names.append(user)
    users[user].append(name)

print("Found " + str(len(user_names)) + " users.")

# Remove users with too few replays
print("Filtering users...")
removed = 0
for user in tqdm(user_names.copy()):
    if (len(users[user]) < MIN_REPLAYS):
        del users[user]
        user_names.remove(user)
        removed += 1

print("Removed " + str(removed) + " users with too few replays.")
print(str(len(user_names)) + " users with enough replays remaining.")

# Function to parse replay file into note sequence
def handle_bsor(name):
    idx = 0
    with open('Z:/beatleader/replays/' + name, 'rb') as f:
        m = make_bsor(f)
    frames = m.frames
    lf = len(frames)

    if (m.info.modifiers != ""): raise Exception()
    if (m.info.leftHanded): raise Exception()
    if (m.info.mode != "Standard"): raise Exception()
    if (m.info.startTime != 0.0): raise Exception()
    if (m.info.failTime != 0.0): raise Exception()
    if (m.info.speed != 0.0): raise Exception()

    def f_before(t):
        nonlocal idx
        if idx >= lf: return None
        while(frames[idx].time < t):
            idx += 1
            if idx >= lf: return None
        idx -= 1
        return frames[idx]

    def f_after(t):
        nonlocal idx
        i = idx
        if i >= lf: return None
        while(frames[i].time < t):
            i += 1
            if i >= lf: return None
        return frames[i]

    def l_interp(x1, x2, y1, y2, x):
        return ((y2 - y1) * x + x2 * y1 - x1 * y2) / (x2 - x1)

    def o_interp(o1, o2, t1, t2, t):
        return [
            l_interp(t1, t2, o1.x, o2.x, t),
            l_interp(t1, t2, o1.y, o2.y, t),
            l_interp(t1, t2, o1.z, o2.z, t),
            l_interp(t1, t2, o1.x_rot, o2.x_rot, t),
            l_interp(t1, t2, o1.y_rot, o2.y_rot, t),
            l_interp(t1, t2, o1.z_rot, o2.z_rot, t),
            l_interp(t1, t2, o1.w_rot, o2.w_rot, t)
        ]

    def f_interp(t):
        f_b = f_before(t)
        f_a = f_after(t)
        if (f_b is None or f_a is None): return [0] * 21
        o = []
        o += o_interp(f_b.head, f_a.head, f_b.time, f_a.time, t)
        o += o_interp(f_b.left_hand, f_a.left_hand, f_b.time, f_a.time, t)
        o += o_interp(f_b.right_hand, f_a.right_hand, f_b.time, f_a.time, t)
        return o

    out = []
    time = 0
    for i in range(900):
        time += 1/30
        out.append(f_interp(time))

    with np.errstate(over='raise'):
        return np.array(out).astype('float16')

# Function to get two replays from different users
different = []
d_combos = []
def getDifferent():
    user1 = np.random.choice(user_names)
    user2 = np.random.choice(user_names)
    if (user1 == user2): return
    if (user1[:-7] == user2): return
    if (user1 == user2[:-7]): return

    rp1 = np.random.choice(users[user1]).strip()
    rp2 = np.random.choice(users[user2]).strip()

    combo = rp1 + '+' + rp2
    alt = rp2 + '+' + rp1
    if (combo in d_combos): return
    if (alt in d_combos): return
    d_combos.append(combo)

    try:
        s1 = handle_bsor(rp1)
        s2 = handle_bsor(rp2)
    except:
        return

    if (np.isnan(s1).any()): return
    if (np.isnan(s2).any()): return
    if (np.isinf(s1).any()): return
    if (np.isinf(s2).any()): return

    different.append([s1, s2])

# Function to get two replays from the same user
same = []
s_combos = []
def getSame():
    user = np.random.choice(user_names)

    rp1 = np.random.choice(users[user]).strip()
    rp2 = np.random.choice(users[user]).strip()
    if (rp1 == rp2): return

    combo = rp1 + '+' + rp2
    alt = rp2 + '+' + rp1
    if (combo in s_combos): return
    if (alt in s_combos): return
    s_combos.append(combo)

    try:
        s1 = handle_bsor(rp1)
        s2 = handle_bsor(rp2)
    except:
        return

    if (np.isnan(s1).any()): return
    if (np.isnan(s2).any()): return
    if (np.isinf(s1).any()): return
    if (np.isinf(s2).any()): return

    same.append([s1, s2])

# Create different-user train set
print("Generating dissimilar replays for training...")
for i in trange(TRAIN_SIZE):
    while (len(different) <= i):
        getDifferent()

diff = np.array(different).astype('float16')
print('Saving result of shape ' + str(diff.shape) + '...')
np.save('./user-similarity/train-false', diff)

del different
del diff
different = []

# Create different-user validation set
print("Generating dissimilar replays for validation...")
for i in trange(VAL_SIZE):
    while (len(different) <= i):
        getDifferent()

diff = np.array(different).astype('float16')
print('Saving result of shape ' + str(diff.shape) + '...')
np.save('./user-similarity/validate-false', diff)

del different
del diff
different = []

# Create different-user test set
print("Generating dissimilar replays for testing...")
for i in trange(TEST_SIZE):
    while (len(different) <= i):
        getDifferent()

diff = np.array(different).astype('float16')
print('Saving result of shape ' + str(diff.shape) + '...')
np.save('./user-similarity/test-false', diff)

del different
del diff

# Create same-user train set
print("Generating similar replays for training...")
for i in trange(TRAIN_SIZE):
    while (len(same) <= i):
        getSame()

similar = np.array(same).astype('float16')
print('Saving result of shape ' + str(similar.shape) + '...')
np.save('./user-similarity/train-true', similar)

del similar
del same
same = []

# Create same-user validation set
print("Generating similar replays for validation...")
for i in trange(VAL_SIZE):
    while (len(same) <= i):
        getSame()

similar = np.array(same).astype('float16')
print('Saving result of shape ' + str(similar.shape) + '...')
np.save('./user-similarity/validate-true', similar)

del similar
del same
same = []

# Create same-user test set
print("Generating similar replays for testing...")
for i in trange(TEST_SIZE):
    while (len(same) <= i):
        getSame()

# Save results
similar = np.array(same).astype('float16')
print('Saving result of shape ' + str(similar.shape) + '...')
np.save('./user-similarity/test-true', similar)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
