# Import libraries
import gc
import numpy as np
from tensorflow import keras
from keras import backend as K

# Configuration options
BATCH_SIZE = 500                        # Batch size for inference

# Load the anonymizer model
anonymizer = keras.models.load_model('../parts/models/anonymizer.keras')

# Load the normalizer model
normalizer = keras.models.load_model('../parts/models/normalizer.keras')

def metaguardplus(data, noise):
    # Anonymize the input data
    noisy = anonymizer.predict([data, noise], batch_size=BATCH_SIZE)

    # Clear session
    K.clear_session()
    gc.collect()

    # Recenter the anonymized data
    data_mean = np.mean(data, axis=(0,1), dtype=np.float64)
    data_std = np.std(data, axis=(0,1), dtype=np.float64)
    anon_mean = np.mean(noisy, axis=(0,1), dtype=np.float64)
    anon_std = np.std(noisy, axis=(0,1), dtype=np.float64)
    noisy = (noisy - anon_mean) / anon_std
    noisy = (noisy * data_std) + data_mean

    # Normalize the anonymized data
    anons = normalizer.predict(noisy, batch_size=BATCH_SIZE)

    # Clip the anonymized data
    anons[:, :, 3:7] = anons[:, :,  3:7].clip(-1, 1)
    anons[:, :,  10:14] = anons[:, :,  10:14].clip(-1, 1)
    anons[:, :,  17:21] = anons[:, :,  17:21].clip(-1, 1)

    return anons
