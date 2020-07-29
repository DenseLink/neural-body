import math

import random
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
import h5py


class benrules_v2:
    # Nested classes for custom datatypes
    # Class that takes the place of a vector.  Used instead of numpy arrays.
    class point:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    # Class to store all the initial and current state properties of a body.
    class body:
        def __init__(self, location, mass, velocity, name=""):
            self.location = location
            self.mass = mass
            self.velocity = velocity
            self.name = name
    # Store the possible bodies as class variable.
    # planet data (location (m), mass (kg), velocity (m/s)
    # Source Data: https://nssdc.gsfc.nasa.gov/planetary/factsheet/
    sun = {"location": point(0, 0, 0), "mass": 1.989e30,
           "velocity": point(0, 0, 0)}
    mercury = {"location": point(0, 57.9e9, 0), "mass": 3.285e23,
               "velocity": point(47400, 0, 0)}
    venus = {"location": point(0, 108.2e9, 0), "mass": 4.87e24,
             "velocity": point(35000, 0, 0)}
    earth = {"location": point(0, 149.6e9, 0), "mass": 5.97e24,
             "velocity": point(29800, 0, 0)}
    mars = {"location": point(0, 227.9e9, 0), "mass": 0.642e24,
            "velocity": point(24100, 0, 0)}
    jupiter = {"location": point(0, 778.6e9, 0), "mass": 1898e24,
               "velocity": point(13100, 0, 0)}
    saturn = {"location": point(0, 1433.5e9, 0), "mass": 568e24,
              "velocity": point(9700, 0, 0)}
    uranus = {"location": point(0, 2872.5e9, 0), "mass": 86.8e24,
              "velocity": point(6835, 0, 0)}
    neptune = {"location": point(0, 4495.1e9, 0), "mass": 102e24,
               "velocity": point(5477, 0, 0)}
    pluto = {"location": point(0, 5906.4e9, 0), "mass": 0.0146e24,
             "velocity": point(4748, 0, 0)}
    sat1 = {
        "location": point(0, 149.602, 0),
        "mass": 4500,
        "velocity": point(7800, 0, 0)
    }

    def __init__(self, time_step, number_of_steps, report_frequency,
                 bodies, additional_bodies):
        """
        Initialize the simulation and create numpy arrays for storing
        simulation history.

        :param time_step:
        :param number_of_steps:
        :param report_freqency:
        :param bodies:
        """
        # Setup initial simulation settings
        self.time_step = time_step
        self.number_of_steps = number_of_steps
        self.report_frequency = report_frequency
        self.bodies = bodies.copy()
        # self.bodies = [
        # self.body(location = self.sun["location"], mass = self.sun["mass"],
        #       velocity = self.sun["velocity"], name = "sun"),
        # self.body(location=self.mercury["location"], mass=self.mercury["mass"],
        #      velocity=self.mercury["velocity"], name="mercury"),
        # self.body(location=self.venus["location"], mass=self.venus["mass"],
        #      velocity=self.venus["velocity"], name="venus"),
        # self.body(location = self.earth["location"], mass = self.earth["mass"],
        #       velocity = self.earth["velocity"], name = "earth"),
        # self.body(location = self.mars["location"], mass = self.mars["mass"],
        #       velocity = self.mars["velocity"], name = "mars"),
        # self.body(location=self.jupiter["location"], mass=self.jupiter["mass"],
        #      velocity=self.jupiter["velocity"], name="jupiter"),
        # self.body(location=self.saturn["location"], mass=self.saturn["mass"],
        #      velocity=self.saturn["velocity"], name="saturn"),
        # self.body(location=self.uranus["location"], mass=self.uranus["mass"],
        #      velocity=self.uranus["velocity"], name="uranus"),
        # self.body(location=self.neptune["location"], mass=self.neptune["mass"],
        #      velocity=self.neptune["velocity"], name="neptune"),
        # self.body(location=self.pluto["location"], mass=self.pluto["mass"],
        #      velocity=self.pluto["velocity"], name="pluto")
        # ]
        # Add satellites and other bodies to the universe.
        self.bodies.extend(additional_bodies)
        # Calculate the number of items that will need to be saved to the
        # caches based on the number of time steps and the reporting frequency.
        self._num_items_in_output = int(number_of_steps // report_frequency) - 1
        # Setup index counter for writing items to output caches
        self._curr_cache_index = -1
        #self.body_locations_hist = []
        # Setup numpy arrays for history tracking
        # Fill arrays with np.nan until the values can be overwritten by the
        # simulation.
        self.acc_np = np.full(
            (self._num_items_in_output, len(self.bodies), 3),
            np.nan,
            dtype=np.float32
        )
        self.vel_np = np.full(
            (self._num_items_in_output, len(self.bodies), 3),
            np.nan,
            dtype=np.float32
        )
        self.pos_np = np.full(
            (self._num_items_in_output, len(self.bodies), 3),
            np.nan,
            dtype=np.float32
        )
        self.dis_np = np.full(
            (self._num_items_in_output, len(self.bodies), 3),
            np.nan,
            dtype=np.float32
        )
        self.mass_np = np.full(
            (len(self.bodies), 1),
            np.nan,
            dtype=np.float32
        )
        # Save the masses for each body.
        for body_index, target_body in enumerate(self.bodies):
            self.mass_np[body_index][0] = target_body.mass
        # Start creating set of numpy arrays that will keep track of current
        # velocities, locations.
        # Numpy arrays will have shape of (num_bodies, 3).
        self.current_vel_np = np.full(
            (len(self.bodies), 3),
            np.nan,
            np.float32
        )
        self.current_loc_np = np.full(
            (len(self.bodies), 3),
            np.nan,
            np.float32
        )
        # Initialize the numpy arrays with the given velocities and locaitons.
        for idx, body in enumerate(self.bodies):
            self.current_vel_np[idx][0] = body.velocity.x
            self.current_vel_np[idx][1] = body.velocity.y
            self.current_vel_np[idx][2] = body.velocity.z
            self.current_loc_np[idx][0] = body.location.x
            self.current_loc_np[idx][1] = body.location.y
            self.current_loc_np[idx][2] = body.location.z


    def run_simulation(self):
        # create output container for each body
        # self.body_locations_hist = []
        # Create initial structure of list whose elements will be a dictionary
        # for each body that has a list for each position dimension that is a
        # time series.
        # for current_body in self.bodies:
        #     self.body_locations_hist.append({"x":[], "y":[], "z":[],
        #                                 "name":current_body.name})

        # Go over each time step and compute the next location after that time
        # step.
        # i keeps track of the current time step in the simulation.
        # pass along report_freq to make sure we write to the history dataframes
        # only as much as needed.  This allows the simulator to run at higher
        # resolution (smaller time step), but the saved data to be lower
        # resolution.
        for i in tqdm(range(1, self.number_of_steps)):
            # Call function to calculate new position after a single time step.
            self._compute_gravity_step(current_step=i)

            # if i % self.report_frequency == 0:
            #     for index, body_location in enumerate(self.body_locations_hist):
            #         body_location["x"].append(self.bodies[index].location.x)
            #         body_location["y"].append(self.bodies[index].location.y)
            #         body_location["z"].append(self.bodies[index].location.z)

    def _compute_gravity_step(self, current_step=0):
        #self._compute_velocity(current_step=current_step)
        self._compute_velocity_vectorized(current_step=current_step)
        #self._update_location(current_step=current_step)
        self._update_location_vectorized(current_step=current_step)

    def _compute_velocity(self, current_step=0):
        for body_index, target_body in enumerate(self.bodies):
            acceleration = self._calculate_single_body_acceleration(body_index)

            target_body.velocity.x += acceleration.x * self.time_step
            target_body.velocity.y += acceleration.y * self.time_step
            target_body.velocity.z += acceleration.z * self.time_step
            # Save the resulting velocity to the velocity history.
            if current_step % self.report_frequency == 0:
                self.vel_np[current_step - 1][body_index][
                    0] = target_body.velocity.x
                self.vel_np[current_step - 1][body_index][
                    1] = target_body.velocity.y
                self.vel_np[current_step - 1][body_index][
                    2] = target_body.velocity.z

                self.acc_np[current_step - 1][body_index][0] = acceleration.x
                self.acc_np[current_step - 1][body_index][1] = acceleration.y
                self.acc_np[current_step - 1][body_index][2] = acceleration.z

    def _compute_velocity_vectorized(self, current_step):
        """
        After getting acceleration from vectorized single_bod_acceleration,
        we can simply multiply by the time step to get velocity.

        :return:
        """
        acceleration_np = self._calc_single_bod_acc_vectorized()
        velocity_np = self.current_vel_np.T.reshape(3, acceleration_np.shape[1], 1) + (acceleration_np * self.time_step)
        # Convert back to the tracking format
        self.current_vel_np[:,:] = velocity_np.T.reshape(acceleration_np.shape[1], 3)
        if current_step % self.report_frequency == 0:
            # Increment the cache index
            self._curr_cache_index += 1
            self.vel_np[self._curr_cache_index, :, :] = self.current_vel_np
            self.acc_np[self._curr_cache_index, :, :] = acceleration_np.T.reshape(acceleration_np.shape[1], 3)

    def _calc_single_bod_acc_vectorized(self):
        """
        This is a prototype version of the acceleration vector adder.  For
        it to work with an arbitrary number of bodies, assume the positions of
        all bodies will be provided as a numpy array with an [x,y,z] vector to
        store the positions.

        :param body_index:
        :return:
        """
        # # To be removed later.  Convert the bodies into a numpy vector with the
        # # position data.
        # # Position vector stores the x,y,z positions of each body.
        # pos_vec = np.full(
        #     (len(self.bodies), 3),
        #     np.nan,
        #     dtype=np.float64
        # )
        # mass_vec = np.full(
        #     (len(self.bodies), 1),
        #     np.nan,
        #     dtype=np.float64
        # )
        # for idx, body in enumerate(self.bodies):
        #     pos_vec[idx][0] = body.location.x
        #     pos_vec[idx][1] = body.location.y
        #     pos_vec[idx][2] = body.location.z
        #     mass_vec[idx][0] = body.mass
        # Create matrix of positions duplicated for later calculating
        # differences between all positions at the same time.
        pos_vec = self.current_loc_np.T.reshape((3, self.current_loc_np.shape[0], 1)) # Have to reshape from previous to get columns that are the x, y, and z dimensions
        pos_mat = pos_vec @ np.ones((1, pos_vec.shape[1]))
        # Find differences between all bodies and all other bodies at
        # the same time.
        diff_mat = pos_mat - pos_mat.transpose((0, 2, 1))
        # Calculate the radius or absolute distances between all bodies
        # and every other body
        r = np.sqrt(np.sum(np.square(diff_mat), axis=0))
        # Calculate the tmp value for every body at the same time
        g_const = 6.67408e-11  # m3 kg-1 s-2
        acceleration_np = g_const * ((diff_mat.transpose((0,2,1)) * (np.reciprocal(r ** 3, out=np.zeros_like(r), where=(r!=0.0))).T) @ self.mass_np)
        return acceleration_np

    def _calculate_single_body_acceleration(self, body_index):
        """
        Calculate the acceleration on the current body given every other body in
        the system.

        :param bodies:
        :param body_index:
        :return:
        """
        G_const = 6.67408e-11  # m3 kg-1 s-2
        acceleration = self.point(0, 0, 0)
        target_body = self.bodies[body_index]
        for index, external_body in enumerate(self.bodies):
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

    def _update_location_vectorized(self, current_step):
        displacement_np = self.current_vel_np * self.time_step
        self.current_loc_np = self.current_loc_np + displacement_np
        if current_step % self.report_frequency == 0:
            # Assume cache index was already incremented in the velocity function.
            self.dis_np[self._curr_cache_index, :, :] = displacement_np
            # Calculate and save position relative to sun.
            sun_pos = self.current_loc_np[0]
            pos_rel_sun_np = self.current_loc_np[:] - sun_pos
            self.pos_np[self._curr_cache_index, :, :] = pos_rel_sun_np


    def _update_location(self, current_step=0):
        """
        Function to update the current positions of the bodies.

        :param bodies:
        :param time_step:
        :return:
        """
        for body_index, target_body in enumerate(self.bodies):
            # Calculate the displacement resulting from the new velocities.
            displacement_x = target_body.velocity.x * self.time_step
            displacement_y = target_body.velocity.y * self.time_step
            displacement_z = target_body.velocity.z * self.time_step
            # Calculate the next location of the body.
            target_body.location.x += displacement_x
            target_body.location.y += displacement_y
            target_body.location.z += displacement_z
            # Save both the displacements and locations to their respective
            # history dataframes.
            if current_step % self.report_frequency == 0:
                self.dis_np[current_step - 1][body_index][0] = displacement_x
                self.dis_np[current_step - 1][body_index][1] = displacement_y
                self.dis_np[current_step - 1][body_index][2] = displacement_z
                # For the current target body, calculate the relative position
                # to the sun and then save to position dataframe.
                # Assume first entry in bodies is always the sun.
                pos_rel_sun_x = target_body.location.x \
                                - self.bodies[0].location.x
                pos_rel_sun_y = target_body.location.y \
                                - self.bodies[0].location.y
                pos_rel_sun_z = target_body.location.z \
                                - self.bodies[0].location.z
                # # Save the relative positions to the dataframe.
                self.pos_np[current_step - 1][body_index][0] = pos_rel_sun_x
                self.pos_np[current_step - 1][body_index][1] = pos_rel_sun_y
                self.pos_np[current_step - 1][body_index][2] = pos_rel_sun_z