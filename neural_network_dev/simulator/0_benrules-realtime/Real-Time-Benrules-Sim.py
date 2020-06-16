'''
Simulator originally from benrules2 on Github.
https://gist.github.com/benrules2/220d56ea6fe9a85a4d762128b11adfba
'''

import math
import random
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from itertools import count   # For keeping track of time step index
from matplotlib.animation import FuncAnimation  # Used for real-time plot update and animation.
plot.style.use('fivethirtyeight')


class point:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

class body:
    def __init__(self, location, mass, velocity, name = ""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name


# Planet data (location (m), mass (kg), velocity (m/s)
sun = {"location":point(0,0,0), "mass":2e30, "velocity":point(0,0,0)}
mercury = {"location":point(0,5.7e10,0), "mass":3.285e23, "velocity":point(47000,0,0)}
venus = {"location":point(0,1.1e11,0), "mass":4.8e24, "velocity":point(35000,0,0)}
earth = {"location":point(0,1.5e11,0), "mass":6e24, "velocity":point(30000,0,0)}
mars = {"location":point(0,2.2e11,0), "mass":2.4e24, "velocity":point(24000,0,0)}
jupiter = {"location":point(0,7.7e11,0), "mass":1e28, "velocity":point(13000,0,0)}
saturn = {"location":point(0,1.4e12,0), "mass":5.7e26, "velocity":point(9000,0,0)}
uranus = {"location":point(0,2.8e12,0), "mass":8.7e25, "velocity":point(6835,0,0)}
neptune = {"location":point(0,4.5e12,0), "mass":1e26, "velocity":point(5477,0,0)}
pluto = {"location":point(0,3.7e12,0), "mass":1.3e22, "velocity":point(4748,0,0)}  # Why is pluto closer than neptune?

# Build list of planets in the simulation, or create your own
bodies = [
    body(location=sun["location"], mass=sun["mass"], velocity=sun["velocity"], name="sun"),
    body(location=mercury["location"], mass=mercury["mass"], velocity=mercury["velocity"], name="mercury"),
    body(location=venus["location"], mass=venus["mass"], velocity=venus["velocity"], name="venus"),
    body(location=earth["location"], mass=earth["mass"], velocity=earth["velocity"], name="earth"),
    body(location=mars["location"], mass=mars["mass"], velocity=mars["velocity"], name="mars"),
    body(location=jupiter["location"], mass=jupiter["mass"], velocity=jupiter["velocity"], name="jupiter"),
    body(location=saturn["location"], mass=saturn["mass"], velocity=saturn["velocity"], name="saturn"),
    body(location=uranus["location"], mass=uranus["mass"], velocity=uranus["velocity"], name="uranus"),
    body(location=neptune["location"], mass=neptune["mass"], velocity=neptune["velocity"], name="neptune"),
    body(location=pluto["location"], mass=pluto["mass"], velocity=pluto["velocity"], name="pluto")
]


def initialize_history():
    """
    Function to initialize the history tracking for all planets during the simulation.
    :return: List with history tracking structure.
    """
    # Create output container for each body
    # Just done once to initialize each body in the list.
    initial_hist = []
    for current_body in bodies:
        initial_hist.append({"x": [], "y": [], "z": [], "name": current_body.name})
    return initial_hist


body_locations_hist = initialize_history()


def calculate_single_body_acceleration(body_index):
    """
    Looks like this is the main function to calculate the acceleration of a single body
    given the location of all current bodies.
    """
    G_const = 6.67408e-11  # m3 kg-1 s-2
    acceleration = point(0,0,0)
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.location.x - external_body.location.x)**2 + (target_body.location.y - external_body.location.y)**2 + (target_body.location.z - external_body.location.z)**2
            r = math.sqrt(r)
            tmp = G_const * external_body.mass / r**3
            acceleration.x += tmp * (external_body.location.x - target_body.location.x)
            acceleration.y += tmp * (external_body.location.y - target_body.location.y)
            acceleration.z += tmp * (external_body.location.z - target_body.location.z)

    return acceleration


def compute_velocity(time_step = 1):
    """
    Calculates the velocity of an object at a.... point in time?
    Is this guy just estimating an acceleration integration with multiplying by time step?
    """
    for body_index, target_body in enumerate(bodies):
        acceleration = calculate_single_body_acceleration(body_index)

        target_body.velocity.x += acceleration.x * time_step
        target_body.velocity.y += acceleration.y * time_step
        target_body.velocity.z += acceleration.z * time_step 


def update_location(time_step = 1):
    """
    Function that moves all body locations forward by one time step.
    :param time_step:
    :return:
    """
    for target_body in bodies:
        target_body.location.x += target_body.velocity.x * time_step
        target_body.location.y += target_body.velocity.y * time_step
        target_body.location.z += target_body.velocity.z * time_step


def compute_gravity_step(time_step = 1):
    """
    Simple function that computes the velocity of each body in each direction
    and updates the body's current location.
    :param time_step:
    :return:
    """
    compute_velocity(time_step = time_step)
    update_location(time_step = time_step)


# def plot_output(bodies, outfile = None):
#     fig = plot.figure()
#     colours = ['r','b','g','y','m','c']
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     max_range = 0
#     for current_body in bodies:
#         max_dim = max(max(current_body["x"]),max(current_body["y"]),max(current_body["z"]))
#         if max_dim > max_range:
#             max_range = max_dim
#         ax.plot(current_body["x"], current_body["y"], current_body["z"], c = random.choice(colours), label = current_body["name"])
#
#     ax.set_xlim([-max_range,max_range])
#     ax.set_ylim([-max_range,max_range])
#     ax.set_zlim([-max_range,max_range])
#     ax.legend()
#
#     if outfile:
#         plot.savefig(outfile)
#     else:
#         plot.show()


def animate(time_step = 1):
    """
    Function to calculate the current position of each planet in the next frame
    and update the live plot with the new frame.
    :return:
    """
    # Given the current list of bodies with their associated current positions, mass, and velocity,
    # Calculate their next positions and velocity.
    compute_gravity_step(time_step)
    # Add to the body history
    for index, body_location in enumerate(body_locations_hist):
        body_location["x"].append(bodies[index].location.x)
        body_location["y"].append(bodies[index].location.y)
        body_location["z"].append(bodies[index].location.z)
    colours = ['r', 'b', 'g', 'y', 'm', 'c']
    # Clear plot axis to prevent color changing and other weird behavior.
    #plot.cla()
    # Plot each body in the current frame.
    #for curr_body in body_locations_hist:
    #    plot.plot(curr_body["x"], curr_body["y"], curr_body["z"], c = random.choice(colours), label = curr_body["name"])
    #plot.tight_layout()
    #plot.show()


# Need to replace run_simulation with a simulation initializer and a function we
# call from matplotlib to live update planet positions for the time step.
# def run_simulation(bodies, names = None, time_step = 1, number_of_steps = 10000, report_freq = 100):
#
#     # Create output container for each body
#     # Just done once to initialize each body in the list.
#     body_locations_hist = []
#     for current_body in bodies:
#         body_locations_hist.append({"x":[], "y":[], "z":[], "name":current_body.name})
#
#     for i in range(1, number_of_steps):
#         # Compute next step in time provided list of bodies and their initial
#         # positions, velocity, and mass.
#         compute_gravity_step(bodies, time_step = 1000)
#
#         if i % report_freq == 0:
#             for index, body_location in enumerate(body_locations_hist):
#                 body_location["x"].append(bodies[index].location.x)
#                 body_location["y"].append(bodies[index].location.y)
#                 body_location["z"].append(bodies[index].location.z)
#
#     return body_locations_hist




if __name__ == "__main__":

    # Set time step (most likely in seconds) in simulation time.
    # How long the simulation steps forward at each iteration.
    time_step = 100
    # Total number of time steps performed by the simulation.
    # Total length of simulation = time_step * number_of_steps.
    number_of_steps = 6000000
    # Frequency at which history is recorded.  Might disregard for real time simulation.
    report_freq = 1000

    # Create index counter that will keep track of the current time step.
    curr_time_step = 0

    # Say screw it and just use text output.
    while curr_time_step < number_of_steps:
        curr_time_step += 1
        # Output the current position of all bodies
        print("-----------------------------------------------------------------")
        for target_body in bodies:
            print("Name: {} / X: {} / Y: {} / Z: {}".format(target_body.name,
                                                            target_body.location.x,
                                                            target_body.location.y,
                                                            target_body.location.z))
        print("-----------------------------------------------------------------")
        # Call the animate function that will calculate the next position and
        animate(time_step)

    # Run the real-time simulation for the specified number of steps.
    #ani = FuncAnimation(plot.gcf(), animate, interval=100000, repeat=True)

    #plot.tight_layout()
    #plot.show()
