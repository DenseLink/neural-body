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
import tensorflow as tf
import os
import h5py
# Imports for multiprocessing producer/consumer data model.
from multiprocessing import Process, Queue, Lock, cpu_count
from concurrent.futures import *


class BenrulesRealTimeSim:
    """
    Class containing a basic, real-time simulator for planet motions that also
    interacts with the NNModelLoader class to load a neural network that
    predicts the motion of one of the bodies in the next time step.

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

        :param x: x position of object in simulation space relative to sun
            at time step 0.
        :param y: y position of object in simulation space relative to sun
            at time step 0.
        :param z: z position of object in simulation space relative to sun
            at time step 0.
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

        :param location: 3D location of body in simulation space represented
            by the _Point class.
        :param mass: Mass in kg of the body.
        :param velocity: Initial velocity magnitude and direction of the body
            at time step 0 in simulation space.  Represented by the
            _Point class.
        :param name: Name of the body being stored.
        """

        def __init__(self, location, mass, velocity, name=""):
            self.location = location
            self.mass = mass
            self.velocity = velocity
            self.name = name

    class _NeuralNet:
        """Class to load Tensorflow model stored in .h5 file and run
        inference with it. """

        def __init__(self, model_path):
            """
            Constructor for model class.  Loads the model into a private
                instance
            variable that can then be called on to make predictions on the
            position of planet the network was trained on.

            :param model_path: Path, including name, to the .h5 file storing
                the neural net.
            :param planet_predicting: Name of planet the model is predicting.
            """

            self._model = tf.keras.models.load_model(model_path)

        def make_prediction(self, input_sequence):
            """
            Function to take a vector of all other planet positions and output
            the XYZ position of the planet being predicted for the current time
            step.

            :param input_vector: Numpy array of all other planets and stars
                in the system.
            :return: Dictionary of X,Y,Z positions of planet we are predicting.
            """

            pred = self._model.predict(input_sequence)
            # Split out the predictions and add a 0 column for the z-axis
            # to the predictions.
            zeroes = np.full(
                (pred.shape[1], 1),
                0.0,
                dtype=np.float64
            )
            pred_displacement = np.append(
                pred[0, :, :2],
                zeroes,
                axis=1
            )
            pred_velocity = np.append(
                pred[0, :, -2:],
                zeroes,
                axis=1
            )
            return pred_displacement, pred_velocity

    # Class Variables

    # Planet data units: (location (m), mass (kg), velocity (m/s)

    def _calc_more_vals(self, count):
        temp_list = []
        for i in range(0, 10):
            count += 1
            temp_list.append(count)
        return count, temp_list

    def _maintain_future_cache(self, output_queue):
        count = 0
        vals = []

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._calc_more_vals, count=count)
            while True:
                if len(vals) < 5 and future.done():
                    count, temp_vals = future.result()
                    vals.extend(temp_vals)
                    future = executor.submit(self._calc_more_vals, count=count)
                if vals:
                    output_queue.put(vals.pop(0))
        #
        # while True:
        #     if len(vals) < 5:
        #         # Launch futures thread
        #         count, temp_vals = self._calc_more_vals(count)
        #         vals.extend(temp_vals)
        #     output_queue.put(vals.pop(0))

    def _parse_sim_config(self, in_df):
        """
        Function to convert

        :param in_df: Dataframe containing the simulation configuration.
        :return: list of Body objects with name, mass, location, and initial
            velocity set.
        """

        # Using iterrows() to go over each row in dataframe and extract info
        # from each row.
        self._bodies = []
        read_planet_pos = []
        read_planet_vel = []
        read_sat_pos = []
        read_sat_vel = []
        read_planet_masses = []
        read_sat_masses = []
        read_planet_names = []
        read_sat_names = []
        for index, row in in_df.iterrows():
            # Check if satellite or other.
            # If satellite, then set predicting name to choose the right
            # neural network.
            if row["satellite?"] == "yes":
                self._satellite_predicting_name = str(row["body_name"])
                read_sat_pos.append(
                    np.array([np.float64(row["location_x"]),
                              np.float64(row["location_y"]),
                              np.float64(row["location_z"])])
                )
                read_sat_vel.append(
                    np.array([np.float64(row["velocity_x"]),
                              np.float64(row["velocity_y"]),
                              np.float64(row["velocity_z"])])
                )
                read_sat_masses.append(
                    np.array([np.float64(row["body_mass"])])
                )
                read_sat_names.append(str(row["body_name"]))
            else:
                read_planet_pos.append(
                    np.array([np.float64(row["location_x"]),
                              np.float64(row["location_y"]),
                              np.float64(row["location_z"])])
                )
                read_planet_vel.append(
                    np.array([np.float64(row["velocity_x"]),
                              np.float64(row["velocity_y"]),
                              np.float64(row["velocity_z"])])
                )
                read_planet_masses.append(
                    np.array([np.float64(row["body_mass"])])
                )
                read_planet_names.append(str(row["body_name"]))

        # Set counters to track the current time step of the simulator and
        # maximum time step the simulator has reached.  This will allow us
        # to rewind the simulator to a previous state and grab coordinates
        # from the dataframe tracking simulation history or to continue
        # simulating time steps that have not been reached yet.
        self._current_time_step = 0
        self._max_time_step_reached = 0

        # Create numpy caches
        # Keep track of how full the data caches are
        self._curr_cache_size = 0
        self._max_cache_size = 100
        # Keep track of time steps in the cache
        self._latest_ts_in_cache = 0
        # Keep track of current position in cache.
        self._curr_cache_index = -1
        # Get numbers of the planets and satellites.
        self._num_planets = len(read_planet_pos)
        self._num_sats = len(read_sat_pos)
        # Create Caches
        self._planet_pos_cache = np.full(
            (self._max_cache_size, self._num_planets, 3),
            np.nan,
            dtype=np.float64
        )
        self._planet_vel_cache = np.full(
            (self._max_cache_size, self._num_planets, 3),
            np.nan,
            dtype=np.float64
        )
        self._sat_pos_cache = np.full(
            (self._max_cache_size, self._num_sats, 3),
            np.nan,
            dtype=np.float64
        )
        self._sat_vel_cache = np.full(
            (self._max_cache_size, self._num_sats, 3),
            np.nan,
            dtype=np.float64
        )
        self._sat_acc_cache = np.full(
            (self._max_cache_size, self._num_sats, 3),
            np.nan,
            dtype=np.float64
        )

        # Initialize the first number of time steps that are equal to the
        # length of input to the LSTM.  Start by reading values into first spot
        # of cache and make those initial values time step 1.
        self._planet_pos_cache[0] = np.stack(read_planet_pos)
        self._planet_vel_cache[0] = np.stack(read_planet_vel)
        self._sat_pos_cache[0] = np.stack(read_sat_pos)
        self._sat_vel_cache[0] = np.stack(read_sat_vel)
        # Merge all body masses into a single numpy array.
        self._masses = np.concatenate(
            (np.stack(read_planet_masses),
             np.stack(read_sat_masses)),
            axis=0
        )
        read_planet_names.extend(read_sat_names)
        self._body_names = read_planet_names
        # # Initialize initial acceleration by calculating acceleration on all
        # # satellites.  Extract last rows as the acceleration for the sats.
        # self._curr_cache_index = 1
        # acceleration_np = self._calc_single_bod_acc_vectorized()
        # acceleration_np = \
        #     acceleration_np.T.reshape(acceleration_np.shape[1], 3)
        # self._curr_cache_index = -1
        # self._sat_acc_cache[0, :, :] = acceleration_np[-self._num_sats:, :]

        # Initialize the acceleration with all 0's for the satellites in the
        # initial time step.
        self._sat_acc_cache[0, :, :] = np.full((self._num_sats, 3), 0,
                                               dtype=np.float64)
        # Update cache trackers
        self._current_time_step += 1
        self._max_time_step_reached += 1
        self._curr_cache_index += 1
        self._latest_ts_in_cache += 1
        self._curr_cache_size += 1
        # Compute the necessary number of gravity steps to fill the LSTM
        # sequence.
        for i in range(1,self._len_lstm_in_seq):
            self._current_time_step += 1
            self._max_time_step_reached += 1
            self._curr_cache_index += 1
            self._compute_gravity_step_vectorized(ignore_nn=True)
            self._latest_ts_in_cache += 1
            self._curr_cache_size += 1

        # Try starting background processes
        self._producer_process = Process(target=self._maintain_future_cache, args=(self._output_queue,))
        self._producer_process.daemon = True
        self._producer_process.start()

    def __init__(self, in_config_df, time_step=800):
        """
        Simulation initialization function.

        :param time_step: Time is seconds between simulation steps.  Used to
            calculate displacement over that time.
        :param planet_predicting: Name of the planet being predicted by the
            neural network.
        :param nn_path: File path to the location of the .h5 file storing the
            neural network that will be loaded with Tensorflow in the
            NeuralNet class.
        """
        # Amount of time that has passed in a single time step in seconds.
        self._time_step = time_step
        # Since we are using an LSTM network, we will need to initialize the
        # the length of the sequence necessary for input into the LSTM.
        self._len_lstm_in_seq = 4
        # Grab info for creating background producer / consumer
        self._num_processes = cpu_count()
        # Create processing queues that producer / consumer will take
        # from and fill.
        buffer_size = 100
        self._output_queue = Queue(buffer_size)
        # Test list to append fake processed values to from the producer.
        self._test_output_list = []
        # Setup the initial set of bodies in the simulation by parsing from
        # config dataframe.
        self._parse_sim_config(in_config_df)
        print("bodies parsed")
        # Grab the current working to use for referencing data files
        self._current_working_directory = \
            os.path.dirname(os.path.realpath(__file__))
        # Create neural network object that lets us run neural network
        # predictions as well.
        # Default to mars model if key in dictionary not found.
        nn_name = 'alpha_lstm_100epochs_mml.h5'
        nn_path = self._current_working_directory + "/nn/" + nn_name
        self._nn = self._NeuralNet(model_path=nn_path)

        # Create archive file to store sim data with necessary datasets and
        # and groups.
        # Setup for incremental resizing and appending.
        self._sim_archive_loc = self._current_working_directory \
                          + '/sim_archives/' \
                          + 'sim_archive.hdf5'
        with h5py.File(self._sim_archive_loc, 'w') as f:
            # Create groups to store planet and sat cache data
            planet_group = f.create_group('planets')
            sat_group = f.create_group('satellites')
            # Create datasets to later append data to.
            planet_group.create_dataset("loc_archive",
                                        (0, self._num_planets, 3),
                                        maxshape=(None, self._num_planets, 3))
            planet_group.create_dataset("vel_archive",
                                        (0, self._num_planets, 3),
                                        maxshape=(None, self._num_planets, 3))
            sat_group.create_dataset("loc_archive",
                                     (0, self._num_sats, 3),
                                     maxshape=(None, self._num_sats, 3))
            sat_group.create_dataset("vel_archive",
                                     (0, self._num_sats, 3),
                                     maxshape=(None, self._num_sats, 3))
            sat_group.create_dataset("acc_archive",
                                     (0, self._num_sats, 3),
                                     maxshape=(None, self._num_sats, 3))
        # Keep track of the latest time step stored in the archive.  Will be
        # used to determine if data from cache actually need flushing.
        self._latest_ts_in_archive = 0

    def _calc_single_bod_acc_vectorized(self):
        """
        This is a prototype version of the acceleration vector adder.  For
        it to work with an arbitrary number of bodies, assume the positions of
        all bodies will be provided as a numpy array with an [x,y,z] vector to
        store the positions.

        :param body_index:
        :return:
        """
        # Combine planets and satellites into a single position vector
        # to calculate all accelerations
        pos_vec = np.concatenate(
            (self._planet_pos_cache[self._curr_cache_index - 1],
             self._sat_pos_cache[self._curr_cache_index - 1]),
            axis=0
        )
        # Have to reshape from previous to get columns that are the x, y,
        # and z dimensions
        pos_vec = pos_vec.T.reshape((3, pos_vec.shape[0], 1))
        pos_mat = pos_vec @ np.ones((1, pos_vec.shape[1]))
        # Find differences between all bodies and all other bodies at
        # the same time.
        diff_mat = pos_mat - pos_mat.transpose((0, 2, 1))
        # Calculate the radius or absolute distances between all bodies
        # and every other body
        r = np.sqrt(np.sum(np.square(diff_mat), axis=0))
        # Calculate the tmp value for every body at the same time
        g_const = 6.67408e-11  # m3 kg-1 s-2
        acceleration_np = g_const * ((diff_mat.transpose((0, 2, 1)) * (
            np.reciprocal(r ** 3, out=np.zeros_like(r),
                          where=(r != 0.0))).T) @ self._masses)
        return acceleration_np

    def _compute_velocity_vectorized(self, ignore_nn=False):
        """
        After getting acceleration from vectorized single_bod_acceleration,
        we can simply multiply by the time step to get velocity.

        :return:
        """
        # Grab the accelerations acting on each body based on current body
        # positions.
        acceleration_np = self._calc_single_bod_acc_vectorized()
        # If not using neural network (like when initializing), combine all
        # bodies into the velocity vector and compute change in velocity.
        if ignore_nn == True:
            velocity_np = np.concatenate(
                (self._planet_vel_cache[self._curr_cache_index - 1],
                 self._sat_vel_cache[self._curr_cache_index - 1]), axis=0
            )
            velocity_np = velocity_np.T.reshape(3, velocity_np.shape[0], 1) \
                          + (acceleration_np * self._time_step)
            # Convert back to caching / tracking format and save to cache
            velocity_np = velocity_np.T.reshape(acceleration_np.shape[1], 3)
            self._planet_vel_cache[self._curr_cache_index, :, :] = \
                velocity_np[:self._num_planets, :]
            self._sat_vel_cache[self._curr_cache_index, :, :] = \
                velocity_np[-self._num_sats:, :]
            acceleration_np = \
                acceleration_np.T.reshape(acceleration_np.shape[1], 3)
            self._sat_acc_cache[self._curr_cache_index, :, :] = \
                acceleration_np[-self._num_sats:]
        else:
            # Compute next state for the planets using the normal simulator.
            velocity_np = self._planet_vel_cache[self._curr_cache_index - 1]
            velocity_np = velocity_np.T.reshape(3, velocity_np.shape[0], 1) \
                          + (acceleration_np[:, 0:self._num_planets, :] * self._time_step)
            # Convert back to caching / tracking format and save to cache
            velocity_np = velocity_np.T.reshape(velocity_np.shape[1], 3)
            self._planet_vel_cache[self._curr_cache_index, :, :] = \
                velocity_np[:self._num_planets, :]
            # Predict next velocity and positions for the
            # Neural network returns multiple time steps after the current
            # time step and includes the next position and velocity.
            # Run inference for each satellite
            acceleration_np = \
                acceleration_np.T.reshape(acceleration_np.shape[1], 3)
            self._sat_acc_cache[self._curr_cache_index, :, :] = \
                acceleration_np[-self._num_sats:]
            # Loop over all satellites.
            for i in range(0, self._num_sats):
                # Extract the input sequence for that satellites' past time
                # steps
                # input_vec = [mass, acc_x, acc_y, vel_x, vel_y] X seq_length
                mass = self._masses[-(self._num_sats - i)]
                acc = self._sat_acc_cache[self._curr_cache_index - self._len_lstm_in_seq:self._curr_cache_index, i, 0:2]
                vel = self._sat_vel_cache[self._curr_cache_index - self._len_lstm_in_seq:self._curr_cache_index, i, 0:2]
                # Repeat the mass for each of the time steps in the sequence.
                mass = np.repeat(mass, self._len_lstm_in_seq, axis=0)
                mass = mass.reshape((-1, 1))
                # Create input and reshape to 3D for the model.
                input_sequence = np.concatenate(
                    [mass, acc, vel],
                    axis=1
                ).reshape(1, self._len_lstm_in_seq, 5)
                # Make prediction of the next n time steps.
                # Output format of [dis_x, dis_y, vel_x, vel_y]
                pred_displacement, pred_velocity = \
                    self._nn.make_prediction(input_sequence)
                # Save the prediction for one time step ahead to the velocity
                # and position caches for the satellites.
                num_ts_into_future = 1
                self._sat_vel_cache[self._curr_cache_index, i, :] = \
                    pred_velocity[num_ts_into_future - 1]
                self._sat_pos_cache[self._curr_cache_index, i, :] = \
                    self._sat_pos_cache[self._curr_cache_index - 1, i, :] \
                    + pred_displacement[num_ts_into_future - 1]

    def _update_location_vectorized(self, ignore_nn = False):
        if ignore_nn == True:
            # Calculate displacement and new location for all bodies
            velocities = np.concatenate(
                (self._planet_vel_cache[self._curr_cache_index],
                 self._sat_vel_cache[self._curr_cache_index]),
                axis=0
            )
            displacement_np = velocities * self._time_step
            # Update locations of planets and satellites
            self._planet_pos_cache[self._curr_cache_index, :, :] = \
                self._planet_pos_cache[self._curr_cache_index - 1, :, :] \
                + displacement_np[:self._num_planets, :]
            self._sat_pos_cache[self._curr_cache_index] = \
                self._sat_pos_cache[self._curr_cache_index - 1, :, :] \
                + displacement_np[-self._num_sats, :]
        else:
            # Only calculate displacement and new positions for the planets.
            # Satellite positions determined in the velocity function.
            velocities = self._planet_vel_cache[self._curr_cache_index]
            displacement_np = velocities * self._time_step
            # Update locations of planets
            self._planet_pos_cache[self._curr_cache_index, :, :] = \
                self._planet_pos_cache[self._curr_cache_index - 1, :, :] \
                + displacement_np

        # Reset all positions relative to the sun.
        # Assume sun is always body 0
        self._planet_pos_cache[self._curr_cache_index, :, :] = \
            self._planet_pos_cache[self._curr_cache_index, :, :] \
            - self._planet_pos_cache[self._curr_cache_index, 0, :]
        self._sat_pos_cache[self._curr_cache_index, :, :] = \
            self._sat_pos_cache[self._curr_cache_index, :, :] \
            - self._planet_pos_cache[self._curr_cache_index, 0, :]

    def _compute_gravity_step_vectorized(self, ignore_nn=False):
        self._compute_velocity_vectorized(ignore_nn=ignore_nn)
        self._update_location_vectorized(ignore_nn=ignore_nn)

    def _flush_cache_to_archive(self):
        """
        If caches are too large, flush to .hdf5 file with groups and datasets

        Returns:
        """

        # Check to make sure there is data in cache that needs to be flushed
        # to the archive before going through the whole file opening
        # difficulty.
        # If the latest TS in the cache is less than the latest time step
        # in the archive, then I need to flush that data to the archive.
        if self._latest_ts_in_cache > self._latest_ts_in_archive:
            # Figure out what data from the cache needs to be added to the
            # archive.
            beg_ts_in_cache = self._latest_ts_in_cache \
                              - self._curr_cache_size + 1
            beg_cache_index = None
            end_cache_index = None
            # Calculate portion of cache that should be flushed
            if self._latest_ts_in_archive in range(beg_ts_in_cache,
                                                   self._latest_ts_in_cache):
                beg_cache_index = \
                    self._curr_cache_index - (
                        self._latest_ts_in_cache
                        - self._latest_ts_in_archive
                    )
                end_cache_index = self._curr_cache_index - 1
            else:
                # Flush the whole cache to the end of the archive
                # Beginning index will be where we stated saving after we
                # filled in the first part of the cache with enough of a
                # sequence to use the LSTM.
                beg_cache_index = 0
                end_cache_index = self._curr_cache_index - 1
            cache_flush_size = end_cache_index - beg_cache_index + 1
            # Open archive for appending, resize datasets, and append current
            # caches to the end of their respective datasets.
            with h5py.File(self._sim_archive_loc, 'a') as f:
                # Get pointers to datasets in archive
                planet_pos_archive = f['planets/loc_archive']
                planet_vel_archive = f['planets/vel_archive']
                sat_pos_archive = f['satellites/loc_archive']
                sat_vel_archive = f['satellites/vel_archive']
                sat_acc_archive = f['satellites/acc_archive']
                # sat_dset = f['satellites/loc_archive']
                # Resize the datasets to accept the new set of cache of data
                planet_pos_archive.resize((planet_pos_archive.shape[0]
                                           + cache_flush_size), axis=0)
                planet_vel_archive.resize((planet_vel_archive.shape[0]
                                           + cache_flush_size), axis=0)
                sat_pos_archive.resize((sat_pos_archive.shape[0]
                                        + cache_flush_size), axis=0)
                sat_vel_archive.resize((sat_vel_archive.shape[0]
                                        + cache_flush_size), axis=0)
                sat_acc_archive.resize((sat_vel_archive.shape[0]
                                        + cache_flush_size), axis=0)
                # Save data to the file
                planet_pos_archive[-cache_flush_size:] = \
                    self._planet_pos_cache[beg_cache_index:end_cache_index + 1]
                planet_vel_archive[-cache_flush_size:] = \
                    self._planet_vel_cache[beg_cache_index:end_cache_index + 1]
                sat_pos_archive[-cache_flush_size:] = \
                    self._sat_pos_cache[beg_cache_index:end_cache_index + 1]
                sat_vel_archive[-cache_flush_size:] = \
                    self._sat_vel_cache[beg_cache_index:end_cache_index + 1]
                sat_acc_archive[-cache_flush_size:] = \
                    self._sat_acc_cache[beg_cache_index:end_cache_index + 1]
                self._latest_ts_in_archive = planet_pos_archive.shape[0]

        # After flushing archive, we need to make sure we always fill the
        # first portion of the cache with enough time steps to make predictions
        # with the LSTM.
        # In normal, forward operation, we can assume we can take the previous
        # time steps from the end of the old cache.  When the time_step is
        # arbitrarily reset to the past, we have to grab the time steps from
        # the archive.  When arbitrarily jumping into the future, the
        # simulation should just run in normal fashion until that time step is
        # reached.
        # We know in normal operation when the curr_cache_index has gone 1
        # beyond the available indices in the cache.
        # Only time this is helpful is when there is no data to fill the rest
        # of the cache with.
        if (self._curr_cache_index == self._max_cache_size) \
                and (self._current_time_step == self._max_time_step_reached):
            # If this is the case, then grab the last time steps from the
            # previous cache extending the length of the LSTM input.
            prev_planet_pos = self._planet_pos_cache[-self._len_lstm_in_seq:]
            prev_planet_vel = self._planet_vel_cache[-self._len_lstm_in_seq:]
            prev_sat_pos = self._sat_pos_cache[-self._len_lstm_in_seq:]
            prev_sat_vel = self._sat_vel_cache[-self._len_lstm_in_seq:]
            prev_sat_acc = self._sat_acc_cache[-self._len_lstm_in_seq:]
            # Fill the first part of the cache with this data.
            self._planet_pos_cache[:self._len_lstm_in_seq] = \
                prev_planet_pos
            self._planet_vel_cache[:self._len_lstm_in_seq] = \
                prev_planet_vel
            self._sat_pos_cache[:self._len_lstm_in_seq] = \
                prev_sat_pos
            self._sat_vel_cache[:self._len_lstm_in_seq] = \
                prev_sat_vel
            self._sat_acc_cache[:self._len_lstm_in_seq] = \
                prev_sat_acc
            # Reset the cache trackers.
            self._latest_ts_in_cache = self._current_time_step - 1
            self._curr_cache_size = self._len_lstm_in_seq
            self._curr_cache_index = self._len_lstm_in_seq
        # If a situation where the current time step has been changed and the
        # cache wasn't just filled up, grab data for previous time steps from
        # the archive.
        else:
            with h5py.File(self._sim_archive_loc, 'r') as f:
                # Get pointers to datasets in archive
                planet_pos_archive = f['planets/loc_archive']
                planet_vel_archive = f['planets/vel_archive']
                sat_pos_archive = f['satellites/loc_archive']
                sat_vel_archive = f['satellites/vel_archive']
                sat_acc_archive = f['satellites/acc_archive']
                # Calculate the indices to extract from the archive.
                beg_archive_index = self._current_time_step \
                    - self._len_lstm_in_seq - 1
                end_archive_index = self._current_time_step - 2
                prev_planet_pos = \
                    planet_pos_archive[beg_archive_index:end_archive_index + 1]
                prev_planet_vel = \
                    planet_vel_archive[beg_archive_index:end_archive_index + 1]
                prev_sat_pos = \
                    sat_pos_archive[beg_archive_index:end_archive_index + 1]
                prev_sat_vel = \
                    sat_vel_archive[beg_archive_index:end_archive_index + 1]
                prev_sat_acc = \
                    sat_acc_archive[beg_archive_index:end_archive_index + 1]
                # Fill the first part of the cache with this data.
                self._planet_pos_cache[:self._len_lstm_in_seq] = \
                    prev_planet_pos
                self._planet_vel_cache[:self._len_lstm_in_seq] = \
                    prev_planet_vel
                self._sat_pos_cache[:self._len_lstm_in_seq] = \
                    prev_sat_pos
                self._sat_vel_cache[:self._len_lstm_in_seq] = \
                    prev_sat_vel
                self._sat_acc_cache[:self._len_lstm_in_seq] = \
                    prev_sat_acc
                # Set the pointer for the current cache index to 1 beyond the
                # past data filled.
                self._curr_cache_index = self._len_lstm_in_seq
                # See if the archive still has some data to fill the cache with
                beg_archive_index = self._current_time_step - 1
                if (self._current_time_step - 1 +
                        (self._max_cache_size - self._len_lstm_in_seq)
                        <= self._latest_ts_in_archive):
                    end_archive_index = self._current_time_step - 1 \
                        + (self._max_cache_size - self._len_lstm_in_seq)
                else:
                    end_archive_index = self._latest_ts_in_archive
                # Fill the latter part of the cache with available data.
                beg_cache_index = self._curr_cache_index
                end_cache_index = beg_cache_index + \
                    (end_archive_index - beg_archive_index)
                self._planet_pos_cache[beg_cache_index:end_cache_index] = \
                    planet_pos_archive[beg_archive_index:end_archive_index]
                self._planet_vel_cache[beg_cache_index:end_cache_index] = \
                    planet_vel_archive[beg_archive_index:end_archive_index]
                self._sat_pos_cache[beg_cache_index:end_cache_index] = \
                    sat_pos_archive[beg_archive_index:end_archive_index]
                self._sat_vel_cache[beg_cache_index:end_cache_index] = \
                    sat_vel_archive[beg_archive_index:end_archive_index]
                self._sat_acc_cache[beg_cache_index:end_cache_index] = \
                    sat_acc_archive[beg_archive_index:end_archive_index]
                # Update the cache size and latest ts in the cache
                self._curr_cache_size = end_cache_index
                self._latest_ts_in_cache = end_archive_index

    def get_next_sim_state_v2(self):
        # We know we need to get positions through calculation and inference
        # rather than from cache when our current simulation has reached
        # the max time step.
        # If the current_time_step-1 == the max time step reached, then we know
        # we have gone beyond the current simulation and need to compute the
        # next simulation time step.
        if self._current_time_step == self._max_time_step_reached:
            # Move current time step forward
            self._current_time_step += 1
            # Move forward the maximum time step the simulation has reached.
            self._max_time_step_reached += 1
            # Move the cache index forward to new place to save calculated
            # data
            self._curr_cache_index += 1
            # Check if cache is full.
            # compute_gravity_step assumes we have available cache for saving
            # current time step and enough previous time steps in cache to
            # feed the LSTM.
            if self._curr_cache_index == self._max_cache_size:
                self._flush_cache_to_archive()
            # Compute and predict next positions of all bodies.
            self._compute_gravity_step_vectorized(ignore_nn=False)
            # Create one numpy array with all body position data to return.
            simulation_positions = np.concatenate(
                (self._planet_pos_cache[self._curr_cache_index],
                 self._sat_pos_cache[self._curr_cache_index]),
                axis=0
            )
            # Update the latest time step stored in the cache
            self._latest_ts_in_cache = self._current_time_step

        # If the current time step is less than the max time step reached and
        # the current time step is in the range of time steps in the cache,
        # then we can go ahead and grab position data from the cache.
        # Be careful to make sure the desired time step also has enough of a
        # data sequence for the neural net to run inference with.
        elif (self._current_time_step < self._max_time_step_reached) and \
                (self._current_time_step in range(
                    self._latest_ts_in_cache - self._curr_cache_size
                    + self._len_lstm_in_seq + 1,
                    self._latest_ts_in_cache +1
                )):
            # If the current time step is in the range of time steps in the
            # cache, we can assume that we can calculate the index in the
            # current cache and use those values for inference.
            beg_cache_ts = self._latest_ts_in_cache \
                           - self._curr_cache_size + 1
            self._curr_cache_index = self._current_time_step - beg_cache_ts
            # Create one numpy array with all body position data to return.
            simulation_positions = np.concatenate(
                (self._planet_pos_cache[self._curr_cache_index],
                 self._sat_pos_cache[self._curr_cache_index]),
                axis=0
            )
            # Advance to the next time step.
            self._current_time_step += 1
            self._curr_cache_index += 1

        # If the current time step is less than the max time step reached and
        # the current time step is NOT in the range of the cache, we need to
        # update the cache before proceeding with getting location information.
        elif (self._current_time_step < self._max_time_step_reached) and \
                (self._current_time_step not in range(
                    self._latest_ts_in_cache - self._curr_cache_size + 1,
                    self._latest_ts_in_cache + 1
                )):
            # Move cache index forward to make it work with the data flushing.
            self._curr_cache_index += 1
            self._flush_cache_to_archive()
            # At this point, the current time step should be loaded into the
            # cache after flushing.  Continue as previous case.
            beg_cache_ts = self._latest_ts_in_cache \
                           - self._curr_cache_size + 1
            self._curr_cache_index = self._current_time_step - beg_cache_ts
            # Create one numpy array with all body position data to return.
            simulation_positions = np.concatenate(
                (self._planet_pos_cache[self._curr_cache_index],
                 self._sat_pos_cache[self._curr_cache_index]),
                axis=0
            )
            # Advance to the next time step.
            self._current_time_step += 1
            self._curr_cache_index += 1

        # As test, try overwriting items from the processing queue
        item = self._output_queue.get()
        simulation_positions[11, 0] = np.float64(self._output_queue.qsize())
        simulation_positions[11, 1] = np.float64(item)

        # Return dictionary with planet name as key and a list with each planet
        # name containing the coordinates
        return simulation_positions

    @property
    def current_time_step(self):
        """
        Getter that retrieves the current time step the simulator is at.

        :return current_time_step: Current time step the simulator is at.
        """
        return self._current_time_step

    @property
    def body_names(self):
        return self._body_names

    @current_time_step.setter
    def current_time_step(self, in_time_step):
        """
        Setter to change the current time step of the simulator.  Essentially
        rewinding the simulation back to a point in its history.

        If negative time entered, default to 0 time.  If time entered past the
        maximum time reached, the simulator will "fast-forward" to that time
        step.
        """

        # Make sure we can't go back before the big bang.
        # Need to keep at least enough time steps for the LSTM network.
        if in_time_step <= self._len_lstm_in_seq:
            in_time_step = self._len_lstm_in_seq + 1
        # If time goes beyond the max time the simulator has reached, advance
        # the simulator to that time.
        if in_time_step > self._max_time_step_reached:
            while self._max_time_step_reached < in_time_step:
                sim_positions = self.get_next_sim_state_v2()
        # If the time is between 0 and the max, set the current time step to 
        # the given time step.
        if (in_time_step >= self._len_lstm_in_seq) and \
                (in_time_step <= self._max_time_step_reached):
            # Update the simulator's time step
            self._current_time_step = in_time_step

    @property
    def max_time_step_reached(self):
        """
        Getter that retrieves the maximum time step the simulation has reached.

        :return max_time_step_reached: Max time step the simulation has
            reached.
        """
        return self._max_time_step_reached
