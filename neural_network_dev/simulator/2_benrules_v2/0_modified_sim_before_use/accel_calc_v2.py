import numpy as np

def calculate_single_body_acceleration(bodies, body_index):
    this_body = bodies[body_index]
    other_bodies = np.concatenate((bodies[:body_index], bodies[body_index+1:]))
    G = 6.67408e-11

    pn = np.tile(this_body[0:3], (len(bodies)-1, 1))
    pm = other_bodies[:, 0:3]
    pd = pm - pn
    d = np.transpose(np.reciprocal(pd, out=np.zeros_like(pd), where=pd!=0))
    r = np.multiply(d, np.absolute(d))

    m = G * this_body[6] * other_bodies[:, 6]

    a = r @ m

test_data = [
    [0, 0, 0, 0, 0, 0, 1e12],
    [50, 0, 0, 1.15534, 0, 0, 1],
    [100, 0, 0, 1.15534, 0, 0, 2],
    [150, 0, 0, 1.49154, 0, 0, 5]
]
test_data = np.array(test_data, dtype=np.double)

