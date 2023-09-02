import numpy as np
from .Bsor import make_bsor

# Function to parse replay file into note sequence
def handle_bsor(name):
    idx = 0
    with open(name, 'rb') as f:
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
        res = np.array(out).astype('float16')

    if (np.isnan(res).any()): raise Exception()
    if (np.isinf(res).any()): raise Exception()
    if (np.any(res > 4)): raise Exception()
    if (np.any(res < -2)): raise Exception()

    return res
