import numpy as np
from tqdm import tqdm

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

def replaysummary(replay):
    summary = []
    for i in range(30):
        chunk = replay[30*i:30+30*i]
        summary.append(chunksummary(chunk))
    return summary

def usersummary(user):
    summary = []
    for replay in user:
        summary.extend(replaysummary(replay))
    return np.array(summary)

def millerfeatures(replays):
    users = []
    for user in tqdm(replays):
        users.append(usersummary(user))
    return np.array(users)
