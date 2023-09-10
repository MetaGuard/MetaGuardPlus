# Import libraries
import random
import string
import os
import numpy as np
from tqdm import tqdm
from common.xror import XROR
from common.parse import handle_bsor
from common.metaguard import metaguard
from common.metaguardplus import metaguardplus

# Define map pool
pool = {
    'acc': [ # Most popular acc maps under 6 stars
        'Hard-Standard-51CB5A9D50F73FEAC9380570CEC13A91D3824E64', # Enemy
        'Hard-Standard-87EFECA37C255E63F8E1EC28C6ABD94A128631C0', # Trauma
        'Hard-Standard-4D03502003602E8E8F0D9F8C84A3E70DA0C389F6', # 99.9
        'Hard-Standard-595F76619E3058472A9DE19DAFCEFFD2B5392093', # Gypsy Tronic
    ],
    'tech': [ # Most popular tech maps from 4 to 8 stars
        'Expert-Standard-43304202EC7681E52B4026313C7AB9099BE2890D', # 666
        'Expert-Standard-B25D5435EF97A85549620125C28A28A153CE396F', # Cheatcode
        'Expert-Standard-4B6672956A0B64FE7922F5DF932077C716D27808', # Cambodia
        'Expert-Standard-816E3A71BB3053B4C1DA3F7A1875C721B2764039', # Time Leaper
    ],
    'speed': [ # Most popular speed/midspeed maps from 8 to 12 stars
        'ExpertPlus-Standard-619250FCF1D3A479D53469B9250837391AE449EA', # Fresh!
        'ExpertPlus-Standard-BA220888BB848D07BDFE3C49A4960E487C63BE82', # SCREW // owo // SCREW
        'ExpertPlus-Standard-27860892DD00DF00DA30A3DB12C23A6999EB853F', # Night Raid
        'ExpertPlus-Standard-64688EE557A6C0916D2108AA3B39DD79C8468DB3', # eden
    ]
}

# Configuration options
MIN_REPLAYS = 100             # Minimum number of replays per user for inclusion

# Shuffle map pool
for cat in ['acc', 'tech', 'speed']:
    np.random.shuffle(pool[cat])

# Get list of replays
with open('../data/user-identification/all-replays.txt') as file:
    replays = file.readlines()

# Remove extensions
replays = [r.split('.')[0] for r in replays]

# Sort by map
print("Sorting replays...")
maps = {}
for r in tqdm(replays):
    map = r.split('-', 1)[-1]
    if (not map in maps): maps[map] = []
    maps[map].append(r)

# Load maps
print("Selecting replays...")
groups = ['control', 'artificial', 'metaguard', 'metaguardplus']
ids = list(range(12))
np.random.shuffle(ids)
n = 0
for i in range(4):
    group = groups[i]
    for cat in ['acc', 'tech', 'speed']:
        map = pool[cat][i]
        id = ids[n]
        n += 1
        np.random.shuffle(maps[map])
        j = 0
        k = 0
        c = random.randint(0, 3)

        # Create output folder
        if not os.path.exists('./usability/replays/' + str(id)):
            os.makedirs('./usability/replays/' + str(id))

        while (j < 4):
            try:
                replay = maps[map][k]
                user = replay.split('-')[0]
                k += 1

                with open('Z:/beatleader/replays/' + replay + '.bsor', 'rb') as f:
                    xror = XROR.fromBSOR(f)

                frames = handle_bsor('Z:/beatleader/replays/' + replay + '.bsor')
                times = np.arange(0, 30, 1/30).reshape(900,1)

                # Remove most existing metadata
                xror.data['info']['software']['activity']['failTime'] = 30
                xror.data['info']['timestamp'] = 0
                xror.data['info']['hardware']['devices'][0]['name'] = 'HMD'
                xror.data['info']['hardware']['devices'][1]['name'] = 'LEFT'
                xror.data['info']['hardware']['devices'][2]['name'] = 'RIGHT'
                xror.data['info']['software']['app']['version'] = '1.X.X'
                xror.data['info']['software']['app']['extensions'][0]['version'] = '0.X.X'
                xror.data['info']['software']['runtime'] = 'oculus'
                xror.data['info']['software']['api'] = 'Oculus'
                xror.data['info']['user']['id'] = '100000000'
                xror.data['info']['user']['name'] = 'Anonymous'

                # Process anonymization if applicable
                if (j == c):
                    if (group == 'metaguard'):
                        frames = metaguard([frames])[0]
                    if (group == 'metaguardplus'):
                        noise = np.random.normal(size=32).astype('float16')
                        frames = metaguardplus(np.repeat([frames], 500, axis=0), np.repeat([noise], 500, axis=0))[0]
                    if (group == 'artificial'):
                        frames = handle_bsor('./usability/artificial/165749-' + map + '.bsor')

                # Freeze motion for first second
                for f in range(30):
                    frames[f] = frames[30]

                xror.data['frames'] = np.hstack([times, frames])

                bsor = xror.toBSOR()
                with open('./usability/replays/' + str(id) + '/' + str(j) + '.bsor', 'wb') as f:
                    f.write(bsor)

                j += 1

            except Exception as e:
                continue

        print(id, group, cat, map, c)
