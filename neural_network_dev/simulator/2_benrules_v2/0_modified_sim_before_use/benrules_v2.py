import math

import random
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc # Garbage collector to free memory every time step.
from numba import jit
import h5py

# Because I'm lazy, just making a global lists to keep track of:
# acceleration
# velocity
# position
# masses

acc_list = []
vel_list = []
pos_list = []
dis_list = []
mass_list = []

acc_np = None
vel_np = None
pos_np = None
dis_np = None
mass_np = None


# Class that takes the place of a vector.  Used instead of numpy arrays.
class point:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z


# Class to store all the initial and current state properties of a body.
class body:
    def __init__(self, location, mass, velocity, name = ""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name


def calculate_single_body_acceleration(bodies, body_index,
                                       current_step = 0, report_freq=1):
    """
    Calculate the acceleration on the current body given every other body in
    the system.

    :param bodies:
    :param body_index:
    :return:
    """
    G_const = 6.67408e-11 #m3 kg-1 s-2
    acceleration = point(0,0,0)
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.location.x - external_body.location.x)**2 \
                + (target_body.location.y - external_body.location.y)**2 \
                + (target_body.location.z - external_body.location.z)**2
            r = math.sqrt(r)
            tmp = G_const * external_body.mass / r**3
            acceleration.x += tmp * (external_body.location.x
                                     - target_body.location.x)
            acceleration.y += tmp * (external_body.location.y
                                     - target_body.location.y)
            acceleration.z += tmp * (external_body.location.z
                                     - target_body.location.z)
    return acceleration


def compute_velocity(bodies, time_step = 1,
                     current_step = 0, report_freq=1):
    for body_index, target_body in enumerate(bodies):
        acceleration = calculate_single_body_acceleration(
            bodies,
            body_index,
            current_step=current_step,
            report_freq=report_freq
        )

        target_body.velocity.x += acceleration.x * time_step
        target_body.velocity.y += acceleration.y * time_step
        target_body.velocity.z += acceleration.z * time_step
        # Save the resulting velocity to the velocity history.
        if current_step % report_freq == 0:
            vel_np[current_step - 1][body_index][0] = target_body.velocity.x
            vel_np[current_step - 1][body_index][1] = target_body.velocity.y
            vel_np[current_step - 1][body_index][2] = target_body.velocity.z

            acc_np[current_step - 1][body_index][0] = acceleration.x
            acc_np[current_step - 1][body_index][1] = acceleration.y
            acc_np[current_step - 1][body_index][2] = acceleration.z


def update_location(bodies, time_step = 1,
                    current_step = 0, report_freq=1):
    """
    Function to update the current positions of the bodies.

    :param bodies:
    :param time_step:
    :return:
    """
    for body_index, target_body in enumerate(bodies):
        # Calculate the displacement resulting from the new velocities.
        displacement_x = target_body.velocity.x * time_step
        displacement_y = target_body.velocity.y * time_step
        displacement_z = target_body.velocity.z * time_step
        # Calculate the next location of the body.
        target_body.location.x += displacement_x
        target_body.location.y += displacement_y
        target_body.location.z += displacement_z
        # Save both the displacements and locations to their respective
        # history dataframes.
        if current_step % report_freq == 0:
            dis_np[current_step - 1][body_index][0] = displacement_x
            dis_np[current_step - 1][body_index][1] = displacement_y
            dis_np[current_step - 1][body_index][2] = displacement_z
            # # For the current target body, calculate the relative position to
            # # the sun and then save to position dataframe.
            # # Assume first entry in bodies is always the sun.
            pos_rel_sun_x = target_body.location.x - bodies[0].location.x
            pos_rel_sun_y = target_body.location.y - bodies[0].location.y
            pos_rel_sun_z = target_body.location.z - bodies[0].location.z
            # # Save the relative positions to the dataframe.
            pos_np[current_step - 1][body_index][0] = pos_rel_sun_x
            pos_np[current_step - 1][body_index][1] = pos_rel_sun_y
            pos_np[current_step - 1][body_index][2] = pos_rel_sun_z


def compute_gravity_step(bodies, time_step = 1,
                         current_step = 0, report_freq=1):
    compute_velocity(bodies, time_step = time_step,
                     current_step = current_step, report_freq=report_freq)
    update_location(bodies, time_step = time_step,
                    current_step = current_step, report_freq=report_freq)


def plot_output(bodies, outfile = None):
    fig = plot.figure()
    colours = ['r','b','g','y','m','c']
    ax = fig.add_subplot(1,1,1, projection='3d')
    max_range = 0
    for current_body in bodies: 
        max_dim = max(max(current_body["x"]),
                      max(current_body["y"]),
                      max(current_body["z"]))
        if max_dim > max_range:
            max_range = max_dim
        ax.plot(current_body["x"],
                current_body["y"],
                current_body["z"],
                c = random.choice(colours),
                label = current_body["name"])
    
    ax.set_xlim([-max_range,max_range])    
    ax.set_ylim([-max_range,max_range])
    ax.set_zlim([-max_range,max_range])
    ax.legend()        

    if outfile:
        plot.savefig(outfile)
    else:
        plot.show()


def run_simulation(bodies, names = None, time_step = 1,
                   number_of_steps = 10000, report_freq = 100):

    #create output container for each body
    body_locations_hist = []
    # Create initial structure of list whose elements will be a dictionary
    # for each body that has a list for each position dimension that is a
    # time series.
    for current_body in bodies:
        body_locations_hist.append({"x":[], "y":[], "z":[],
                                    "name":current_body.name})

    # Go over each time step and compute the next location after that time
    # step.
    # i keeps track of the current time step in the simulation.
    # pass along report_freq to make sure we write to the history dataframes
    # only as much as needed.  This allows the simulator to run at higher
    # resolution (smaller time step), but the saved data to be lower
    # resolution.
    for i in tqdm(range(1,number_of_steps)):
        # Call function to calculate new position after a single time step.
        compute_gravity_step(bodies, time_step = time_step, current_step=i,
                             report_freq=report_freq)

        #gc.collect()
        # if i % report_freq == 0:
        #     for index, body_location in enumerate(body_locations_hist):
        #         body_location["x"].append(bodies[index].location.x)
        #         body_location["y"].append(bodies[index].location.y)
        #         body_location["z"].append(bodies[index].location.z)

    return body_locations_hist        


#planet data (location (m), mass (kg), velocity (m/s)
# Source Data: https://nssdc.gsfc.nasa.gov/planetary/factsheet/
sun = {"location":point(0,0,0), "mass":1.989e30, "velocity":point(0,0,0)}
mercury = {"location":point(0,57.9e9,0), "mass":3.285e23, "velocity":point(47400,0,0)}
venus = {"location":point(0,108.2e9,0), "mass":4.87e24, "velocity":point(35000,0,0)}
earth = {"location":point(0,149.6e9,0), "mass":5.97e24, "velocity":point(29800,0,0)}
mars = {"location":point(0,227.9e9,0), "mass":0.642e24, "velocity":point(24100,0,0)}
jupiter = {"location":point(0,778.6e9,0), "mass":1898e24, "velocity":point(13100,0,0)}
saturn = {"location":point(0,1433.5e9,0), "mass":568e24, "velocity":point(9700,0,0)}
uranus = {"location":point(0,2872.5e9,0), "mass":86.8e24, "velocity":point(6835,0,0)}
neptune = {"location":point(0,4495.1e9,0), "mass":102e24, "velocity":point(5477,0,0)}
pluto = {"location":point(0,5906.4e9,0), "mass":0.0146e24, "velocity":point(4748,0,0)}
sat1 = {
    "location": point(0, 149.602, 0),
    "mass": 4500,
    "velocity": point(7800, 0, 0)
}

if __name__ == "__main__":

    #build list of planets in the simulation, or create your own
    bodies = [
        body(location = sun["location"], mass = sun["mass"],
              velocity = sun["velocity"], name = "sun"),
        body(location=mercury["location"], mass=mercury["mass"],
             velocity=mercury["velocity"], name="mercury"),
        body(location=venus["location"], mass=venus["mass"],
             velocity=venus["velocity"], name="venus"),
        body(location = earth["location"], mass = earth["mass"],
              velocity = earth["velocity"], name = "earth"),
        body(location = mars["location"], mass = mars["mass"],
              velocity = mars["velocity"], name = "mars"),
        body(location=jupiter["location"], mass=jupiter["mass"],
             velocity=jupiter["velocity"], name="jupiter"),
        body(location=saturn["location"], mass=saturn["mass"],
             velocity=saturn["velocity"], name="saturn"),
        body(location=uranus["location"], mass=uranus["mass"],
             velocity=uranus["velocity"], name="uranus"),
        body(location=neptune["location"], mass=neptune["mass"],
             velocity=neptune["velocity"], name="neptune"),
        body(location=pluto["location"], mass=pluto["mass"],
             velocity=pluto["velocity"], name="pluto")
        #body(location=sat1["location"], mass=sat1["mass"],
        #     velocity=sat1["velocity"], name="sat1"),
    ]

    """
    Run the simulation.  
    -> time_step is the length of the simulation time step in seconds.
    -> number_of_steps is the number of time steps to run the simulation for.
    -> report_freq is how often to save the results.  Set to 1 to write a 
        result at every time step.
    
    Units of time to seconds:
    365.25 days = 31557600 seconds
    100 years = 3155760000 seconds
    248 years (orbit of pluto) = 7826284800
    
    """
    # Setup simulation settings
    time_step = 800
    number_of_steps = 9782856
    report_frequency = 1

    # Set the shape of and initialize global numpy arrays that store the
    # simulation history.
    # Fill arrays with np.nan until the values can be overwritten by the
    # simulation.
    acc_np = np.full(
        (number_of_steps-1, len(bodies), 3),
        np.nan,
        dtype=np.float32
    )
    vel_np = np.full(
        (number_of_steps-1, len(bodies), 3),
        np.nan,
        dtype=np.float32
    )
    pos_np = np.full(
        (number_of_steps-1, len(bodies), 3),
        np.nan,
        dtype=np.float32
    )
    dis_np = np.full(
        (number_of_steps-1, len(bodies), 3),
        np.nan,
        dtype=np.float32
    )
    mass_np = np.full(
        (len(bodies), 1),
        np.nan,
        dtype=np.float32
    )

    # Save the masses for each body.
    for body_index, target_body in enumerate(bodies):
        mass_np[body_index][0] = target_body.mass

    motions = run_simulation(
        bodies,
        time_step=time_step,
        number_of_steps=number_of_steps,
        report_freq=report_frequency
    )
    #plot_output(motions, outfile = 'orbits.png')

    print("Simulation Complete")
    # Save the resulting numpy arrays
    results_dir = 'output/'
    # np.save(results_dir + 'a.npy', acc_np)
    # np.save(results_dir + 'v.npy', vel_np)
    # np.save(results_dir + 'd.npy', pos_np)
    # np.save(results_dir + 'p.npy', dis_np)
    np.save(results_dir + 'm.npy', mass_np)

    # Save results as hdf5 for later partial reading.
    with h5py.File(results_dir + 'a.hdf5', 'w') as f:
        dset = f.create_dataset("acc", data=acc_np)
    with h5py.File(results_dir + 'v.hdf5', 'w') as f:
        dset = f.create_dataset("vel", data=vel_np)
    with h5py.File(results_dir + 'p.hdf5', 'w') as f:
        dset = f.create_dataset("pos", data=pos_np)
    with h5py.File(results_dir + 'd.hdf5', 'w') as f:
        dset = f.create_dataset("dis", data=dis_np)
    print('Results Saved')