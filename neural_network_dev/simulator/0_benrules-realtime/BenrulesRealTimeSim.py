"""
This module contains the real time simulator class that
takes a set of bodies and calculates their next state / frame / time step
in a simulation.

"""

# Imports
import math
import pandas as pd
import numpy as np
from NNModelLoader import NeuralNet


class BenrulesRealTimeSim:
    # Sub classes for simulation data types
    class _point:
        """
        Class to represent a 3D point in space in a location list.
        """
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class _body:
        """
        Class to represent physical attributes of a body.
        """
        def __init__(self, location, mass, velocity, name=""):
            self.location = location
            self.mass = mass
            self.velocity = velocity
            self.name = name

    # Class Variables
    # Planet data (location (m), mass (kg), velocity (m/s)
    sun = {"location": _point(0, 0, 0), "mass": 2e30, "velocity": _point(0, 0, 0)}
    mercury = {"location": _point(0, 5.7e10, 0), "mass": 3.285e23, "velocity": _point(47000, 0, 0)}
    venus = {"location": _point(0, 1.1e11, 0), "mass": 4.8e24, "velocity": _point(35000, 0, 0)}
    earth = {"location": _point(0, 1.5e11, 0), "mass": 6e24, "velocity": _point(30000, 0, 0)}
    mars = {"location": _point(0, 2.2e11, 0), "mass": 2.4e24, "velocity": _point(24000, 0, 0)}
    jupiter = {"location": _point(0, 7.7e11, 0), "mass": 1e28, "velocity": _point(13000, 0, 0)}
    saturn = {"location": _point(0, 1.4e12, 0), "mass": 5.7e26, "velocity": _point(9000, 0, 0)}
    uranus = {"location": _point(0, 2.8e12, 0), "mass": 8.7e25, "velocity": _point(6835, 0, 0)}
    neptune = {"location": _point(0, 4.5e12, 0), "mass": 1e26, "velocity": _point(5477, 0, 0)}
    pluto = {"location": _point(0, 3.7e12, 0), "mass": 1.3e22,
             "velocity": _point(4748, 0, 0)}  # Why is pluto closer than neptune?

    def _initialize_history(self):
        # Create list of columns
        history_columns = []
        for current_body in self._bodies:
            history_columns.append(current_body.name + "_x")
            history_columns.append(current_body.name + "_y")
            history_columns.append(current_body.name + "_z")
        # Create dataframe with above column names for tracking history.
        initial_df = pd.DataFrame(columns=history_columns)
        # Return the empty stucture of the dataframe.
        return initial_df

    def __init__(self, time_step=100, planet_predicting='pluto', nn_path=''):
        """
        Initialize the history list that keeps track of past planet positions.
        Will use Pandas dataframe that can easily have portions converted to
        numpy arrays.
        """
        # Setup the initial set of bodies in the simulation.
        self._bodies = [
            self._body(location=self.sun["location"], mass=self.sun["mass"], velocity=self.sun["velocity"], name="sun"),
            self._body(location=self.mercury["location"], mass=self.mercury["mass"], velocity=self.mercury["velocity"], name="mercury"),
            self._body(location=self.venus["location"], mass=self.venus["mass"], velocity=self.venus["velocity"], name="venus"),
            self._body(location=self.earth["location"], mass=self.earth["mass"], velocity=self.earth["velocity"], name="earth"),
            self._body(location=self.mars["location"], mass=self.mars["mass"], velocity=self.mars["velocity"], name="mars"),
            self._body(location=self.jupiter["location"], mass=self.jupiter["mass"], velocity=self.jupiter["velocity"], name="jupiter"),
            self._body(location=self.saturn["location"], mass=self.saturn["mass"], velocity=self.saturn["velocity"], name="saturn"),
            self._body(location=self.uranus["location"], mass=self.uranus["mass"], velocity=self.uranus["velocity"], name="uranus"),
            self._body(location=self.neptune["location"], mass=self.neptune["mass"], velocity=self.neptune["velocity"], name="neptune"),
            self._body(location=self.pluto["location"], mass=self.pluto["mass"], velocity=self.pluto["velocity"], name="pluto")
        ]
        # Setup pandas dataframe to keep track of simulation history.
        # Pandas dataframe is easy to convert to any data file format and has plotting shortcuts
        # for easier end-of-simulation plotting.
        self._body_locations_hist = self._initialize_history()
        # Amount of time that has passed in a single time step (I think in seconds)
        self._time_step = time_step
        # Create neural network object that lets us run neural network predictions as well.
        self._nn = NeuralNet(model_path=nn_path,
                             planet_predicting=planet_predicting)
        # Add current system state to the history tracking.
        coordinate_list = []
        for target_body in self._bodies:
            coordinate_list.append(target_body.location.x)
            coordinate_list.append(target_body.location.y)
            coordinate_list.append(target_body.location.z)
        # Store coordinates to dataframe tracking simulation history
        self._body_locations_hist.loc[len(self._body_locations_hist)] = coordinate_list

    def _calculate_single_body_acceleration(self, body_index):
        """
        Looks like this is the main function to calculate the acceleration of a single body
        given the location of all current bodies.
        """
        G_const = 6.67408e-11  # m3 kg-1 s-2
        acceleration = self._point(0, 0, 0)
        target_body = self._bodies[body_index]
        for index, external_body in enumerate(self._bodies):
            if index != body_index:
                r = (target_body.location.x - external_body.location.x) ** 2 + (
                            target_body.location.y - external_body.location.y) ** 2 + (
                                target_body.location.z - external_body.location.z) ** 2
                r = math.sqrt(r)
                tmp = G_const * external_body.mass / r ** 3
                acceleration.x += tmp * (external_body.location.x - target_body.location.x)
                acceleration.y += tmp * (external_body.location.y - target_body.location.y)
                acceleration.z += tmp * (external_body.location.z - target_body.location.z)

        return acceleration

    def _compute_velocity(self):
        """
        Calculates the velocity of an object at a.... point in time?
        Is this guy just estimating an acceleration integration with multiplying by time step?
        """
        for body_index, target_body in enumerate(self._bodies):
            acceleration = self._calculate_single_body_acceleration(body_index)
            target_body.velocity.x += acceleration.x * self._time_step
            target_body.velocity.y += acceleration.y * self._time_step
            target_body.velocity.z += acceleration.z * self._time_step

    def _update_location(self):
        """
        Function that moves all body locations forward by one time step.
        :param:
        :return:
        """
        for target_body in self._bodies:
            target_body.location.x += target_body.velocity.x * self._time_step
            target_body.location.y += target_body.velocity.y * self._time_step
            target_body.location.z += target_body.velocity.z * self._time_step

    def _compute_gravity_step(self):
        """
        Simple function that computes the velocity of each body in each direction
        and updates the body's current location.
        :param time_step:
        :return: dictionary with 3 item lists containing the x,y,z positions of each planet.
                The dictionary is referenced by the planet name.
        """
        self._compute_velocity()
        self._update_location()

    def get_next_sim_state(self):
        """
        Function to calculate the position of all system bodies in the next time step / frame.
        Also stores history of the simulation to self._body_locations_hist.
        :return:
        """
        # Predict planet location using neural network.
        # Strangely, its actually faster to append to a normal python list than a
        # numpy array, so better to aggregate with list then convert to numpy array.
        # Get position data from last point in simulation.  Use as input vector to nn.
        #input_vector = np.array(coordinate_list[0:-3]).reshape(1, -1)
        input_vector = self._body_locations_hist.iloc[len(self._body_locations_hist)-1, 0:-3].values.reshape(1, -1)
        pred_pos = self._nn.make_prediction(input_vector)

        # Compute the next time step and update positions of all bodies in self_bodies.
        self._compute_gravity_step()
        # Format position data for each planet into simple lists.  Dictionary key is the name
        # of the planet.
        simulation_positions = {}
        # Also create a coordinate list that can be added as row to the history dataframe
        coordinate_list = []
        for target_body in self._bodies:
            simulation_positions.update({target_body.name: [target_body.location.x,
                                                            target_body.location.y,
                                                            target_body.location.z]})
            coordinate_list.append(target_body.location.x)
            coordinate_list.append(target_body.location.y)
            coordinate_list.append(target_body.location.z)
        # Store coordinates to dataframe tracking simulation history
        self._body_locations_hist.loc[len(self._body_locations_hist)] = coordinate_list

        # Return dictionary with planet name as key and a list with each planet name
        # containing the coordinates
        return simulation_positions, pred_pos

    @property
    def body_locations_hist(self):
        return self._body_locations_hist

    @property
    def bodies(self):
        return self._bodies