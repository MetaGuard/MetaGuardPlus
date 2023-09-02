# Import libraries
import random
import string
import os
import numpy as np
from common.xror import XROR
from common.Bsor import make_bsor
from common.parse import handle_bsor

# Define map pool
pool = {
    'acc': [ # Most popular acc maps under 4 stars
        'Hard-Standard-51CB5A9D50F73FEAC9380570CEC13A91D3824E64', # Enemy
        'Hard-Standard-83475886CE251C12F1C1755D15A2FE494776AE93', # me & u
        'Hard-Standard-6BE290E33FD748CD678D4C38F1D59E582A44045F', # Introduction
        'Hard-Standard-87EFECA37C255E63F8E1EC28C6ABD94A128631C0', # Trauma
    ],
    'tech': [ # Most popular tech maps from 4 to 8 stars
        'Expert-Standard-43304202EC7681E52B4026313C7AB9099BE2890D', # 666
        'Expert-Standard-3C37D1264FC2754051B4045AC84909CDF0703D62', # Tribal Trial
        'Expert-Standard-EDDCA14F0E90F5DA5A2337B8DD2B1EB381771400', # Deimos
        'Expert-Standard-8E6BC0438A81C60AEEC4E9AFB937145FC9ECD947', # Chakra
    ],
    'speed': [ # Most popular speed/midspeed maps from 8 to 12 stars
        'ExpertPlus-Standard-619250FCF1D3A479D53469B9250837391AE449EA', # Fresh!
        'ExpertPlus-Standard-BA220888BB848D07BDFE3C49A4960E487C63BE82', # SCREW // owo // SCREW
        'ExpertPlus-Standard-4D03502003602E8E8F0D9F8C84A3E70DA0C389F6', # 99.9
        'ExpertPlus-Standard-64688EE557A6C0916D2108AA3B39DD79C8468DB3', # eden
    ]
}

# Shuffle map pool
for cat in ['acc', 'tech', 'speed']:
    np.random.shuffle(pool[cat])

# Get list of replays
with open('../data/user-identification/all-replays.txt') as file:
    replays = file.readlines()

# Remove extensions
replays = [r.split('.')[0] for r in replays]

# Sort by map
maps = {}
for r in replays:
    map = r.split('-', 1)[-1]
    if (not map in maps): maps[map] = []
    maps[map].append(r)

# Load maps
groups = ['control', 'artificial', 'metaguard', 'metaguardplus']
ids = []
for i in range(4):
    group = groups[i]
    for cat in ['acc', 'tech', 'speed']:
        map = pool[cat][i]
        id = ''.join([random.choice(string.ascii_letters) for i in range(5)])
        ids.append(id)
        os.makedirs('./usability/replays/' + id)
        np.random.shuffle(maps[map])
        j = 0
        k = 0
        while (j < 4):
            try:
                replay = maps[map][k]
                k += 1

                with open('Z:/beatleader/replays/' + replay + '.bsor', 'rb') as f:
                    xror = XROR.fromBSOR(f)

                frames = handle_bsor('Z:/beatleader/replays/' + replay + '.bsor')
                times = np.arange(0, 30, 1/30).reshape(900,1)
                xror.data['info']['software']['activity']['failTime'] = 30
                xror.data['frames'] = np.hstack([times, frames])

                bsor = xror.toBSOR()
                with open('./usability/replays/' + id + '/' + str(j), 'wb') as f:
                    f.write(bsor)

                j += 1

            except:
                continue

        print(id, group, cat, map)

np.random.shuffle(ids)
with open('./usability/groups.txt', 'w') as f:
    f.write(",".join(ids))
