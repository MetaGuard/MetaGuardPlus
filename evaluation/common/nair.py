import numpy as np
from tqdm import tqdm
import math

def axissummary(chunk):
    mean = np.mean(chunk)
    median = np.median(chunk)
    min = np.mean(chunk)
    max = np.median(chunk)
    std = np.std(chunk)
    return [mean, median, min, max, std]

def chunksummary(chunk):
    details = []
    for i in range(21):
        details.extend(axissummary(chunk[:,i]))
    return details

def replaysummary(frames, notes):
    summary = []
    for note in notes:
        if (note[0] != 0 and note[1] > 0.05 and note[1] < 29.95):
            id, time = note[0], note[1]
            frame = math.floor(time*30)
            result = [id]
            result.extend(chunksummary(frames[max(0,frame-30):frame]))
            result.extend(chunksummary(frames[frame:min(900,frame+30)]))
            summary.append(result)
    return summary

def usersummary(user):
    summary = []
    replays, notes = user

    for i in range(len(replays)):
        summary.extend(replaysummary(replays[i], notes[i]))

    return np.vstack([np.array(summary)[:300], np.array(summary)[-300:]])

def nairfeatures(replays, notes):
    users = []
    for user in tqdm(list(zip(replays, notes))):
        users.append(usersummary(user))
    return np.array(users)
