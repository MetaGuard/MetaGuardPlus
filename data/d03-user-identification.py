import os
import time
import numpy as np
from tqdm import tqdm, trange
from bsor.Bsor import make_bsor

# Configuration options
REPLAY_DIR = "Z:/beatleader/replays/"   # Directory with replays
MIN_REPLAYS = 20                        # Minimum number of replays per user for inclusion
MAX_REPLAYS = 80                        # Maximum number of replays per user for inclusion
NUM_USERS = 5000                        # Number of users to include
NUM_REPLAYS = 20                        # Number of replays per user to include

# Start measuring performance
start_time = time.time()

# Get list of replays in directory
print("Scanning replay directory...")
replays = list(os.listdir(REPLAY_DIR))
print("Found " + str(len(replays)) + " replays.")

# Save replay list as .txt
f = open("./user-identification/all-replays.txt", "w")
f.writelines([r + "\n" for r in replays])
f.close()

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
    if (len(users[user]) < MIN_REPLAYS or len(users[user]) > MAX_REPLAYS):
        del users[user]
        user_names.remove(user)
        removed += 1

print("Removed " + str(removed) + " users.")
print(str(len(user_names)) + " users remaining.")

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

    notes = []
    for note in m.notes:
        if note.event_time < 30:
            notes.append([
                note.note_id,
                note.event_time
            ])

    notes = notes + ([[0,0]] * 100)
    notes = notes[:100]

    with np.errstate(over='raise'):
        res = np.array(out).astype('float16')

    if (np.isnan(res).any()): raise Exception()
    if (np.isinf(res).any()): raise Exception()
    if (np.any(res > 4)): raise Exception()
    if (np.any(res < -2)): raise Exception()

    with np.errstate(over='raise'):
        return res, np.array(notes).astype('float16')

# Function to get two replays from different users
output_motion = []
output_notes = []
o_users = []
def getUser():
    user = np.random.choice(user_names)
    if user in o_users: return
    o_users.append(user)

    replays = []
    notes = []
    np.random.shuffle(users[user])
    k = -1

    for j in trange(NUM_REPLAYS, leave=False):
        while (len(replays) <= j):
            k += 1
            if (k >= len(users[user])): return;
            if (k >= 10 and len(replays) == 0): return;
            if (k >= 30 and len(replays) < 5): return;
            if (len(users[user]) - k < NUM_REPLAYS - len(replays)): return;

            rp = users[user][k]

            try:
                s, n = handle_bsor(rp)
            except:
                continue

            if (np.isnan(s).any()): continue
            if (np.isinf(s).any()): continue
            if (np.isnan(n).any()): continue
            if (np.isinf(n).any()): continue

            replays.append(s)
            notes.append(n)

    output_motion.append(replays)
    output_notes.append(notes)

# Create user replay set
print("Processing replays...")
for i in trange(NUM_USERS):
    while (len(output_motion) <= i):
        getUser()

# Save results
out = np.array(output_motion).astype('float16')
print('Saving motion of shape ' + str(out.shape) + '...')
np.save('./user-identification/user-replays', out)

out = np.array(output_notes).astype('float16')
print('Saving notes of shape ' + str(out.shape) + '...')
np.save('./user-identification/user-notes', out)

# Log performance results
end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
