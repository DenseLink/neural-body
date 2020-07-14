import math

import random
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Because I'm lazy, just making a global pandas dataframes for the:
# acceleration
# velocity
# position
# masses
acc_df = pd.DataFrame(
    columns=[
        'time_step',
        'body_name',
        'acc_x',
        'acc_y',
        'acc_z'
    ]
)
vel_df = pd.DataFrame(
    columns=[
        'time_step',
        'body_name',
        'vel_x',
        'vel_y',
        'vel_z'
    ]
)
# Positions will be calculated relative to the sun
pos_df = pd.DataFrame(
    columns=[
        'time_step',
        'body_name',
        'pos_x',
        'pos_y',
        'pos_z'
    ]
)
# Keep track of resulting displacements from the new velocity and accelerations.
dis_df = pd.DataFrame(
    columns=[
        'time_step',
        'body_name',
        'dis_x',
        'dis_y',
        'dis_z'
    ]
)

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

    # Save the resulting acceleration for the current time step.
    # Saving with same reporting frequency as other histories.
    if current_step % report_freq == 0:
        history_entry = [
            current_step,
            bodies[body_index].name,
            acceleration.x,
            acceleration.y,
            acceleration.z
        ]
        # Append new row to the pandas dataframe.
        acc_df.loc[len(acc_df)] = history_entry
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
            history_entry = [
                current_step,
                bodies[body_index].name,
                target_body.velocity.x,
                target_body.velocity.y,
                target_body.velocity.z
            ]
            # Append new row to the pandas dataframe.
            vel_df.loc[len(vel_df)] = history_entry


def update_location(bodies, time_step = 1,
                    current_step = 0, report_freq=1):
    """
    Function to update the current positions of the bodies.

    :param bodies:
    :param time_step:
    :return:
    """
    for target_body in bodies:
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
            dis_hist_row = [
                current_step,
                target_body.name,
                displacement_x,
                displacement_y,
                displacement_z
            ]
            dis_df.loc[len(dis_df)] = dis_hist_row
            # For the current target body, calculate the relative position to
            # the sun and then save to position dataframe.
            # Assume first entry in bodies is always the sun.
            pos_rel_sun_x = target_body.location.x - bodies[0].location.x
            pos_rel_sun_y = target_body.location.y - bodies[0].location.y
            pos_rel_sun_z = target_body.location.z - bodies[0].location.z
            # Save the relative positions to the dataframe.
            pos_his_row = [
                current_step,
                target_body.name,
                pos_rel_sun_x,
                pos_rel_sun_y,
                pos_rel_sun_z
            ]
            pos_df.loc[len(pos_df)] = pos_his_row

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
    for i in range(1,number_of_steps):
        # Call function to calculate new position after a single time step.
        compute_gravity_step(bodies, time_step = time_step, current_step=i,
                             report_freq=report_freq)
        
        if i % report_freq == 0:
            for index, body_location in enumerate(body_locations_hist):
                body_location["x"].append(bodies[index].location.x)
                body_location["y"].append(bodies[index].location.y)           
                body_location["z"].append(bodies[index].location.z)       

    return body_locations_hist        
            
#planet data (location (m), mass (kg), velocity (m/s)
sun = {"location":point(0,0,0), "mass":2e30, "velocity":point(0,0,0)}
mercury = {"location":point(0,5.7e10,0), "mass":3.285e23, "velocity":point(47000,0,0)}
venus = {"location":point(0,1.1e11,0), "mass":4.8e24, "velocity":point(35000,0,0)}
earth = {"location":point(0,1.5e11,0), "mass":6e24, "velocity":point(30000,0,0)}
mars = {"location":point(0,2.2e11,0), "mass":2.4e24, "velocity":point(24000,0,0)}
jupiter = {"location":point(0,7.7e11,0), "mass":1e28, "velocity":point(13000,0,0)}
saturn = {"location":point(0,1.4e12,0), "mass":5.7e26, "velocity":point(9000,0,0)}
uranus = {"location":point(0,2.8e12,0), "mass":8.7e25, "velocity":point(6835,0,0)}
neptune = {"location":point(0,4.5e12,0), "mass":1e26, "velocity":point(5477,0,0)}
pluto = {"location":point(0,3.7e12,0), "mass":1.3e22, "velocity":point(4748,0,0)}

if __name__ == "__main__":

    #build list of planets in the simulation, or create your own
    bodies = [
        body( location = sun["location"], mass = sun["mass"], velocity = sun["velocity"], name = "sun"),
        body( location = earth["location"], mass = earth["mass"], velocity = earth["velocity"], name = "earth"),
        body( location = mars["location"], mass = mars["mass"], velocity = mars["velocity"], name = "mars"),
        body( location = venus["location"], mass = venus["mass"], velocity = venus["velocity"], name = "venus"),
        ]
    
    motions = run_simulation(bodies, time_step = 100, number_of_steps = 80000, report_freq = 1000)
    plot_output(motions, outfile = 'orbits.png')

    # Convert the lists of dictionaries tracking simulator history to pandas dataframe.
    # Appending to pandas dataframe was unbelievably slow.
