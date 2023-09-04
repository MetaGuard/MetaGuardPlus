# Import libraries
import numpy as np
from diffprivlib.mechanisms import LaplaceBoundedDomain
from tqdm import tqdm

# Bounded Laplace Mechanism (https://arxiv.org/pdf/1808.10410.pdf)
def BoundedLaplace(x, epsilon, lower, upper):
    LBD = LaplaceBoundedDomain(
        epsilon=epsilon,
        delta=0,
        sensitivity=(upper-lower),
        lower=lower,
        upper=upper
    )
    return LBD.randomise(x)

# Original MetaGuard (https://arxiv.org/pdf/2208.05604.pdf)
# Medium Setting, Height, Wingspan, & Room Transforms
def transform(replay):
    height = np.median(replay[:,1]).clip(1.496, 1.826)
    wingspan = np.max(np.abs(replay[:,14] - replay[:,7])).clip(1.556, 1.899)
    room_width = np.max(np.abs(replay[:,0])).clip(0.1, 1)
    room_height = np.max(np.abs(replay[:,2])).clip(0.1, 1)

    height_f = BoundedLaplace(height, 3, 1.496, 1.826)
    wingspan_f = BoundedLaplace(wingspan, 1, 1.556, 1.899)
    room_width_f = BoundedLaplace(room_width, 1, 0.1, 1)
    room_height_f = BoundedLaplace(room_height, 1, 0.1, 1)

    replay[:,1] = replay[:,1] * (height_f / height)
    replay[:,7] = replay[:,7] * (wingspan_f / wingspan)
    replay[:,14] = replay[:,14] * (wingspan_f / wingspan)
    replay[:,0] = replay[:,0] * (room_width_f / room_width)
    replay[:,2] = replay[:,2] * (room_height_f / room_height)

    return replay

def metaguard(replays):
    return [transform(replay) for replay in tqdm(replays)]
