"""
    A 2-D simulation with 100 particles, whose initial conditions are generated
    randomly according to a Gaussian distribution; velocity is always
    tangential to each particle's position relative to the origin.

    In the origin, there is a mass much larger than the others.

    –– INPUT ARGUMENTS ––––––––––––––––––––––––––––––––––––––––––––––––

        ARG     DESCRIPTION     SHAPE       S.I. UNITS

        x0  –   positions   –   (N,p)   –   meters
        v0  –   velocities  –   (N,p)   –   meters/second
        w0  –   ang. vel.   –   (N,p)   –   radians/second
        m   –   masses      –   (N,)    –   kilograms
        q   –   charges     –   (N,)    –   coulombs
        r   –   radii       –   (N,)    –   meters
"""
from nbody import Sphere, spheres, animate, save
#from ..nbody import Sphere, spheres, animate, save
import numpy as np

# Filename for saving results
filename = "orbits"

N = 100

x = (0,80)
v = (0.6,0.5)
w = (0,0)
m = (5,50)
q = (0,0)
r = (1,0.5)

N = int(N)

# Set dimension of the simulation
p = 2

# Setting up the particle positions
# Draw random positions from a normal distribution.
# (loc=center of the distribution, scale=std deviation of distribution,
# size=output shape of the array of random values)
# loc=0, scale=80, size=(100,2) (shape)
x = np.random.normal(x[0], x[1], (N,p))

# Getting the position's unit vectors
# Get the magnitude of the vector and use to create a unit vector for position.
x_norm = np.linalg.norm(x, axis = 0)
# Make unit vectors of the position vectors for each body.
x_unit = x/x_norm

# Setting the velocities to be orthogonal to the position vectors
# Get absolute magnitude of velocity.
v_abs = np.random.normal(v[0], v[1], N)[:,None]  # Get random samples from a gaussian distribution.
# Create velocity unit vectors that are normal to the position vectors
# Matrix used is a counter-clockwise rotation of the unit vector by 90 degrees.
v_unit = x @ np.array([[0, -1],[1, 0]])
# Amplify the velocity unit vectors by the randomly selected velocity magnitude (speed)
v = v_abs * v_unit

# Setting up the particle angular velocities
w = np.random.normal(w[0], w[1], (N,1))

# Creating mass array, and accounting for negative values
m = np.random.normal(m[0], m[1], (N,1))
m[m < 0] = np.abs(m[m < 0])
m[m == 0] = 1

# Creating charge array
q = np.random.normal(q[0], q[1], (N,1))

# Creating radius array, and accounting for negative values
r = np.random.normal(r[0], r[1], (N,1))
r[r < 0] = np.abs(r[r < 0])
r[r == 0] = 1

# Creating a new Sphere() object
x2 = (0, 0)
v2 = (0, 0)
w2 = 0
m2, q2, r2 = 1E11, 0, 10
P1 = Sphere(x2, v2, w2, m2, q2, r2)

T = 50
dt = 0.1

# Creating an instance of class System
S = spheres(x, v, w, m, q, r)

# Adding the new particle to the System
S.add(P1)

# Solving for the given T and dt
S.solve(T, dt, collision = True)

# Saving the results to file
save(S, filename)

# Saving an animation of the system
# S.animate()
#animate(S, filename)
