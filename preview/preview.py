# Import libraries
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from os import listdir
import sys
sys.path.append("..")
from evaluation.common.xror import XROR
from evaluation.common.parse import handle_bsor

# Load the trained models
anonymizer = keras.models.load_model('../parts/models/anonymizer.keras')
normalizer = keras.models.load_model('../parts/models/normalizer.keras')

# Anonymize one replay
def anonymize(replay):
    with open('./0-replays/' + replay, 'rb') as f:
        xror = XROR.fromBSOR(f)

    frames = handle_bsor('./0-replays/' + replay)
    times = np.arange(0, 30, 1/30).reshape(900,1)
    xror.data['info']['software']['activity']['failTime'] = 30
    xror.data['info']['user']['id'] = '100000000'
    xror.data['info']['user']['name'] = 'Anonymous'
    xror.data['frames'] = np.hstack([times, frames])

    bsor = xror.toBSOR()
    with open('./1-control/' + replay, 'wb') as f:
        f.write(bsor)

    noise = np.random.normal(size=32).astype('float16')
    anons = anonymizer.predict([np.array([frames]), np.array([noise])])[0]

    data_mean = np.mean(frames[30:], axis=0, dtype=np.float64)
    data_std = np.std(frames[30:], axis=0, dtype=np.float64)
    anon_mean = np.mean(anons[30:], axis=0, dtype=np.float64)
    anon_std = np.std(anons[30:], axis=0, dtype=np.float64)

    anon_norm = (anons - anon_mean) / anon_std
    anon_norm = (anon_norm * data_std) + data_mean
    anons = anon_norm.copy()

    anons[:, 3:7] = anons[:, 3:7].clip(-1, 1)
    anons[:, 10:14] = anons[:, 10:14].clip(-1, 1)
    anons[:, 17:21] = anons[:, 17:21].clip(-1, 1)
    xror.data['frames'] = np.hstack([times, anons])

    bsor = xror.toBSOR()
    with open('./2-anonymized/' + replay, 'wb') as f:
        f.write(bsor)

    anons = normalizer(np.array([anon_norm])).numpy()[0]
    anons[:, 3:7] = anons[:, 3:7].clip(-1, 1)
    anons[:, 10:14] = anons[:, 10:14].clip(-1, 1)
    anons[:, 17:21] = anons[:, 17:21].clip(-1, 1)

    xror.data['frames'] = np.hstack([times, anons])

    bsor = xror.toBSOR()
    with open('./3-normalized/' + replay, 'wb') as f:
        f.write(bsor)

# Anonymize all replays
for file in tqdm(list(listdir('./0-replays'))):
    anonymize(file)
