"""Real-Time basic simulator for planetary motions with neural network
inference for prediction of pluto's position.

This module contains the BenrulesRealTimeSim class, which creates a real time
simulator of the sun, planets, and pluto.  The non-real-time version was
forked from GitHub user benrules2 at the below repo:
https://gist.github.com/benrules2/220d56ea6fe9a85a4d762128b11adfba
The simulator originally would simulate a fixed number of time steps and
then output a record of the past simulation.  The code was repackaged into a
class and methods added to allow querying and advancing of the simulator
in real-time at a fixed time step.
Further additions were made to then integrate a class for loading a neural
network (NeuralNet) that would load a Tensorflow model, take a vector
containing all other planetary positions (X, Y, Z) and output the predicted
position of Pluto in the next time step.
"""

# Imports
import math
import pandas as pd
import numpy as np
from NNModelLoader import NeuralNet


class BenrulesRealTimeSim:
    """
    Class containing a basic, real-time simulator for planet motions that also
    interacts with the NNModelLoader class to load a neural network that
    predicts the motion of one of the bodies in the next time step.

    Attributes:
        sun         Initial physical state of the Sun in the simulation
        mercury     Initial physical state of the Mercury in the simulation
        venus       Initial physical state of the Venus in the simulation
        earth       Initial physical state of the Earth in the simulation
        mars        Initial physical state of the Mars in the simulation
        jupiter     Initial physical state of the Jupiter in the simulation
        saturn      Initial physical state of the Saturn in the simulation
        uranus      Initial physical state of the Uranus in the simulation
        neptune     Initial physical state of the Neptune in the simulation
        pluto       Initial physical state of the Pluto in the simulation

    Instance Variables:
    :ivar _bodies: Current physical state of each body at the current time step
    :ivar _body_locations_hist: Pandas dataframe containing the positional
    history of all bodies in the simulation.
    :ivar _time_step: The amount of time the simulation uses between time
    steps.  The amount of "simulation time" that passes.
    :ivar _nn: NNModelLoader object instance that contains the neural network
    loaded in Tensorflow.

    """
    # Nested Classes
    class _Point:
        """
        Class to represent a 3D point in space in a location list.

        The class can be used to represent a fixed point in 3D space or the
        magnitude and direction of a velocity or acceleration vector in 3D
        space.

        :param x: x position of object in simulation space relative to sun at
        time step 0.
        :param y: y position of object in simulation space relative to sun at
        time step 0.
        :param z: z position of object in simulation space relative to sun at
        time step 0.
        """

        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class _Body:
        """
        Class to represent physical attributes of a body.

        This class stores the location (from the point class), mass, velocity,
        and name associated with a body in simulation space.

        :param location: 3D location of body in simulation space represented by
        the _Point class.
        :param mass: Mass in kg of the body.
        :param velocity: Initial velocity magnitude and direction of the body
        at time step 0 in simulation space.  Represented by the _Point class.
        :param name: Name of the body being stored.
        """

        def __init__(self, location, mass, velocity, name=""):
            self.location = location
            self.mass = mass
            self.velocity = velocity
            self.name = name

    # Class Variables

    # Planet data units: (location (m), mass (kg), velocity (m/s)
    #
    # These are the possible bodies that can be simulated with stable orbits
    # over long time periods.  Other bodies can be added later if needed,
    # but initial attempts at satellites led to unstable orbits.
    sun = {"location": _Point(0, 0, 0),
           "mass": 2e30,
           "velocity": _Point(0, 0, 0)}
    mercury = {"location": _Point(0, 5.7e10, 0),
               "mass": 3.285e23,
               "velocity": _Point(47000, 0, 0)}
    venus = {"location": _Point(0, 1.1e11, 0),
             "mass": 4.8e24,
             "velocity": _Point(35000, 0, 0)}
    earth = {"location": _Point(0, 1.5e11, 0),
             "mass": 6e24,
             "velocity": _Point(30000, 0, 0)}
    mars = {"location": _Point(0, 2.2e11, 0),
            "mass": 2.4e24,
            "velocity": _Point(24000, 0, 0)}
    jupiter = {"location": _Point(0, 7.7e11, 0),
               "mass": 1e28,
               "velocity": _Point(13000, 0, 0)}
    saturn = {"location": _Point(0, 1.4e12, 0),
              "mass": 5.7e26,
              "velocity": _Point(9000, 0, 0)}
    uranus = {"location": _Point(0, 2.8e12, 0),
              "mass": 8.7e25,
              "velocity": _Point(6835, 0, 0)}
    neptune = {"location": _Point(0, 4.5e12, 0),
               "mass": 1e26,
               "velocity": _Point(5477, 0, 0)}
    pluto = {"location": _Point(0, 3.7e12, 0),
             "mass": 1.3e22,
             "velocity": _Point(4748, 0, 0)}

    def _initialize_history(self):
        """
        Function to create the initial structure of a Pandas dataframe for
        recording the position of every body in simulation space at each time
        step.

        :return: Pandas dataframe containing the structure for recording
        entire history of the simulation.
        """

        # Create list of columns
        history_columns = []
        for current_body in self._bodies:
            history_columns.append(current_body.name + "_x")
            history_columns.append(current_body.name + "_y")
            history_columns.append(current_body.name + "_z")
        # Create dataframe with above column names for tracking history.
        initial_df = pd.DataFrame(columns=history_columns)
        # Return the empty structure of the dataframe.
        return initial_df

    def __init__(self, time_step=100, planet_predicting='pluto', nn_path=''):
        """
        Simulation initialization function.

        :param time_step: Time is seconds between simulation steps.  Used to
        displacement over that time.
        :param planet_predicting: Name of the planet being predicted by the
        neural network.
        :param nn_path: File path to the location of the .h5 file storing the
        neural network that will be loaded with Tensorflow in the NeuralNet
        class.
        """

        # Setup the initial set of bodies in the simulation.
        self._bodies = [
            self._Body(location=self.sun["location"],
                       mass=self.sun["mass"],
                       velocity=self.sun["velocity"],
                       name="sun"),
            self._Body(location=self.mercury["location"],
                       mass=self.mercury["mass"],
                       velocity=self.mercury["velocity"],
                       name="mercury"),
            self._Body(location=self.venus["location"],
                       mass=self.venus["mass"],
                       velocity=self.venus["velocity"],
                       name="venus"),
            self._Body(location=self.earth["location"],
                       mass=self.earth["mass"],
                       velocity=self.earth["velocity"],
                       name="earth"),
            self._Body(location=self.mars["location"],
                       mass=self.mars["mass"],
                       velocity=self.mars["velocity"],
                       name="mars"),
            self._Body(location=self.jupiter["location"],
                       mass=self.jupiter["mass"],
                       velocity=self.jupiter["velocity"],
                       name="jupiter"),
            self._Body(location=self.saturn["location"],
                       mass=self.saturn["mass"],
                       velocity=self.saturn["velocity"],
                       name="saturn"),
            self._Body(location=self.uranus["location"],
                       mass=self.uranus["mass"],
                       velocity=self.uranus["velocity"],
                       name="uranus"),
            self._Body(location=self.neptune["location"],
                       mass=self.neptune["mass"],
                       velocity=self.neptune["velocity"],
                       name="neptune"),
            self._Body(location=self.pluto["location"],
                       mass=self.pluto["mass"],
                       velocity=self.pluto["velocity"],
                       name="pluto")
        ]
        # CONTINUE DOCUMENTATION HERE.
        # Setup pandas dataframe to keep track of simulation history.
        #
        # Pandas dataframe is easy to convert to any data file format
        # and has plotting shortcuts for easier end-of-simulation plotting.
        self._body_locations_hist = self._initialize_history()
        # Amount of time that has passed in a single time step
        # (I think in seconds)
        self._time_step = time_step
        # Create neural network object that lets us run neural network
        # predictions as well.
        self._nn = NeuralNet(model_path=nn_path,
                             planet_predicting=planet_predicting)
        self._planet_predicting_name = planet_predicting
        # Add current system state to the history tracking.
        coordinate_list = []
        for target_body in self._bodies:
            coordinate_list.append(target_body.location.x)
            coordinate_list.append(target_body.location.y)
            coordinate_list.append(target_body.location.z)
        # Store coordinates to dataframe tracking simulation history
        self._body_locations_hist.loc[len(self._body_locations_hist)] \
            = coordinate_list

    def _calculate_single_body_acceleration(self, body_index):
        """
        Function to calculate the acceleration forces on a given body.

        This function takes in the index of a particular body in the class'
        bodies list and calculates the resulting acceleration vector on that
        body given the physical state of all other bodies.

        :param body_index: Index of body in class' body list on which the
        resulting acceleration will be calculated.
        """

        G_const = 6.67408e-11  # m3 kg-1 s-2
        acceleration = self._Point(0, 0, 0)
        target_body = self._bodies[body_index]
        for index, external_body in enumerate(self._bodies):
            if index != body_index:
                r = (target_body.location.x - external_body.location.x) ** 2 \
                    + (target_body.location.y - external_body.location.y) ** 2 \
                    + (target_body.location.z - external_body.location.z) ** 2
                r = math.sqrt(r)
                tmp = G_const * external_body.mass / r ** 3
                acceleration.x += tmp * (external_body.location.x
                                         - target_body.location.x)
                acceleration.y += tmp * (external_body.location.y
                                         - target_body.location.y)
                acceleration.z += tmp * (external_body.location.z
                                         - target_body.location.z)

        return acceleration

    def _compute_velocity(self):
        """
        Calculates the velocity vector for each body in the class' bodies list.

        Given the physical state of each body in the system, this function
        calls the _calculate_single_body_acceleration on each body in the
        system and uses the resulting acceleration vector along with the
        defined simulation time step to calculate the resulting velocity
        vector for each body.
        """

        for body_index, target_body in enumerate(self._bodies):
            acceleration = self._calculate_single_body_acceleration(body_index)
            target_body.velocity.x += acceleration.x * self._time_step
            target_body.velocity.y += acceleration.y * self._time_step
            target_body.velocity.z += acceleration.z * self._time_step

    def _update_location(self):
        """
        Calculates next location of each body in the system.

        This method, assuming the _compute_velocity method was already called,
        takes the new velocities of all bodies and uses the defined time step
        to calculate the resulting displacement for each body over that time
        step.  The displacement is then added to the current positions in order
        to get the body's new location.
        """

        for target_body in self._bodies:
            target_body.location.x += target_body.velocity.x * self._time_step
            target_body.location.y += target_body.velocity.y * self._time_step
            target_body.location.z += target_body.velocity.z * self._time_step

    def _compute_gravity_step(self):
        """
        Calls the _compute_velocity and _update_location methods in order to
        update the system state by one time step.
        """

        self._compute_velocity()
        self._update_location()

    def get_next_sim_state(self):
        """
        Function to calculate the position of all system bodies in the next
        time step.

        When this method is called, the current system state is passed to the
        neural network to calculate the position of a certain body in the next
        time step.  After the neural network completes, the simulation then
        advances all bodies ahead using "physics".  The positions of all
        bodies resulting from the "physics" are then packaged into a dictionary
        with the body name as key and a list containing the x,y,z coordinates
        of the body as the value attached to that key.  The predicted position
        from the neural network is also packaged as a dictionary with the name
        as key and predicted coordinates as the value.

        :returns:
            - simulation_positions - Dictionary containing all body positions
            in the next time step calculated with "physics".
            - pred_pos - Dictionary containing the predicted position of a
            body using the neural network.
        """

        # Predict planet location using neural network.
        #
        # Strangely, its actually faster to append to a normal python list
        # than a numpy array, so better to aggregate with list then convert
        # to numpy array.
        # Get position data from last point in simulation.
        # Use as input vector to nn.
        #
        # Need to use the name of the planet to find which one to extract from
        # the input vector.

        # Extract last row of dataframe recording simulator history, remove
        # the planet we are trying to predict from the columns, and convert
        # to numpy array as the input vector to the neural network.
        last_row = self._body_locations_hist.iloc[-1, :].copy()
        # Drop columns from dataframe for the planet we are trying to predict.
        last_row = last_row.drop([self._planet_predicting_name + "_x",
                                  self._planet_predicting_name + "_y",
                                  self._planet_predicting_name + "_z"])
        input_vector = last_row.values.reshape(1, -1)

        # OLD CONVERSION FROM BEFORE - REMOVE LATER
        # input_vector = self._body_locations_hist.iloc[
        #                len(self._body_locations_hist)-1,
        #                0:-3].values.reshape(1, -1)
        pred_pos = self._nn.make_prediction(input_vector)

        # Compute the next time step and update positions of all bodies
        # in self_bodies.
        self._compute_gravity_step()
        # Format position data for each planet into simple lists.  Dictionary
        # key is the name of the planet.
        simulation_positions = {}
        # Also create a coordinate list that can be added as row to the
        # history dataframe
        coordinate_list = []
        for target_body in self._bodies:
            simulation_positions.update(
                {target_body.name: [target_body.location.x,
                                    target_body.location.y,
                                    target_body.location.z]})
            coordinate_list.append(target_body.location.x)
            coordinate_list.append(target_body.location.y)
            coordinate_list.append(target_body.location.z)
        # Store coordinates to dataframe tracking simulation history
        self._body_locations_hist.loc[
            len(self._body_locations_hist)] = coordinate_list

        # Return dictionary with planet name as key and a list with each planet
        # name containing the coordinates
        return simulation_positions, pred_pos

    @property
    def body_locations_hist(self):
        """
        Getter that returns a Pandas dataframe with the entire simulation
        history.

        :return body_locations_hist: Pandas dataframe containing the entire
        history of the simulation.  The positional data of all bodies over all
        time steps.
        """
        return self._body_locations_hist

    @property
    def bodies(self):
        """
        Getter that retrieves the current state of the entire system in the
        simulation.

        :return bodies:  Returns the list of bodies.  Each item in the list
        is a Body object containing the physical state of the body.
        """
        return self._bodies

    @property
    def planet_predicting_name(self):
        """
        Getter that retrieves the name of the planet the neural network is
        trying to predict the position of.

        :return planet_predicting_name:  Name of the planet the neural network
        is trying to predict.
        """
        return self._planet_predicting_name