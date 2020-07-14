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

# Set the number of bodies in the simulation.
N = 1
N = int(N)

# Setting up parameters for data generators for each simulation specification.
# Can probably get rid of this later.
x = (0, 80)
v = (0.6, 0.5)
w = (0, 0)
m = (5, 50)
q = (0, 0)
r = (1, 0.5)
# Set geometric dimension for the simulation.
p = 2

# Initial simulation data from https://nssdc.gsfc.nasa.gov/planetary/factsheet/
# Using average distance from sun, but will need to update later with
# positions accounting for elliptical orbits, the Perihelion and Aphelion.

# Create list of body names
body_names = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jumpiter',
              'saturn', 'uranus', 'neptun', 'pluto']

# Setting up the particle positions
# Use 0.1 as small, close to 0 quantity so it can later
# be used in division to calculate unit vectors.
x = np.array(
    [
        #[57.9e9, 0],
        #[108.2e9, 0],
        [149.6e9, 0.00000001]
        #[227.9e9, 0]
        #[778.6e9, 0],
        #[1433.5e9, 0],
        #[2872.5e9, 0],
        #[4495.1e9, 0],
        #[5906.4e9, 0]
    ]
)

# Setting velocities and making them perpendicular to the position
# Create array containing the velocity magnitudes.
v = np.array(
    [
        #[0, 47.9e3],
        #[0, 35.0e3],
        [0, 29.8e3]
        #[0, 24.1e3]
        #[0, -13.1e3],
        #[0, -9.7e3],
        #[0, -6.8e3],
        #[0, -5.4e3],
        #[0, -4.7e3]
    ]
)

# Setting up the particle angular velocities
# Set to all zeros
w = np.random.normal(w[0], w[1], (N,1))

# Setting up body masses
m = np.array(
    [
        #[1.989e30],
        #[0.3302e24],
        #[4.8690e24],
        [5.9740e24]
        #[2.4e24]#[0.6419e24]
        #[1898e24],
        #[568e24],
        #[86.8e24],
        #[102e24],
        #[0.0146e24],
    ]
)

# Creating charge array
q = np.random.normal(q[0], q[1], (N,1))

# Setting up body radii
r = np.array(
    [
        #[(1.3927e9) / 2],
        #[(4879e3) / 2],
        #[(12104e3) / 2],
        [(12756e3) / 2],
        #[(6792e3) / 2]
        #[(142984e3) / 2],
        #[(120536e3) / 2],
        #[(51118e3) / 2],
        #[(49528e3) / 2],
        #[(2370e3) / 2]
    ]
)

# Creating a new Sphere() object for the Sun
x2 = (0, 0)
v2 = (0, 0)
w2 = 0
m2, q2, r2 = 1.989e30, 0, (1.3927e9 / 2)
P1 = Sphere(x2, v2, w2, m2, q2, r2)

T = 2
dt = 1

# Creating an instance of class System
#S = spheres(x, v, w, m, q, r)
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
