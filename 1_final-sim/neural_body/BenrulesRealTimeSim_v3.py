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
from multiprocessing import Process, Queue, Lock, cpu_count, Value
from concurrent.futures import *
import concurrent.futures.thread
from threading import Thread

import time


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
    @staticmethod
    def _future_calc_single_bod_acc_vectorized(planet_pos,
                                               sat_pos,
                                               masses):
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
            (planet_pos,
             sat_pos),
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
                          where=(r != 0.0))).T) @ masses)
        return acceleration_np

    def _future_compute_new_pos_vectorized(self, planet_pos, planet_vel,
                                           sat_pos, sat_vel, sat_acc, masses,
                                           time_step, neural_net,
                                           num_in_items_seq_lstm,
                                           num_out_seq_lstm,
                                           ignore_nn=False):
        """
        After getting acceleration from vectorized single_bod_acceleration,
        we can simply multiply by the time step to get velocity.

        :return:
        """

        # Get the number of planets and satellites
        num_planets = planet_pos.shape[0]
        num_sats = sat_pos.shape[0]
        # num_in_items_seq_lstm = sat_vel.shape[0]
        # num_out_seq_lstm = 10

        if ignore_nn == True:
            # If not using neural network, compute everything as many times as
            # we should to get the same amount in the output as we would if
            # we used the nn.  Combining all bodies together.
            #Initialize values in lists.  Drop the first item later.
            new_planet_pos = [planet_pos]
            new_planet_vel = [planet_vel]
            new_sat_pos = [sat_pos]
            new_sat_vel = [sat_vel[-1]]
            new_sat_acc = [sat_acc[-1]]
            # Loop over the next i time steps and add to the "new" lists
            for i in range(0, num_out_seq_lstm):
                # Grab the accelerations acting on each body based on current
                # body positions.
                acceleration_np = self._future_calc_single_bod_acc_vectorized(
                    planet_pos=new_planet_pos[-1],
                    sat_pos=new_sat_pos[-1],
                    masses=masses
                )
                # Initialize the velocity matrix
                velocity_np = np.concatenate(
                    (new_planet_vel[-1],
                     new_sat_vel[-1]), axis=0
                )
                # Calculate the new valocities
                velocity_np = velocity_np.T.reshape(3, velocity_np.shape[0], 1) \
                              + (acceleration_np * time_step)
                # Convert back to caching / tracking format and save to cache
                velocity_np = velocity_np.T.reshape(acceleration_np.shape[1],
                                                    3)
                new_planet_vel.append(velocity_np[:num_planets, :])
                new_sat_vel.append(velocity_np[-num_sats:, :])
                acceleration_np = \
                    acceleration_np.T.reshape(acceleration_np.shape[1], 3)
                new_sat_acc.append(acceleration_np[-num_sats:])
                # Calculate displacement and new location for all bodies
                # Displacement is based on the current velocity.
                velocities = np.concatenate(
                    (new_planet_vel[-1],
                     new_sat_vel[-1]),
                    axis=0
                )
                displacement_np = velocities * time_step
                # Update new positions of planets and satellites
                new_planet_pos.append(
                    new_planet_pos[-1] + displacement_np[:num_planets, :]
                )
                new_sat_pos.append(
                    new_sat_pos[-1] + displacement_np[-num_sats:, :]
                )
                # Set all positions relative to the sun assumed to be at index 0
                new_planet_pos[-1] = new_planet_pos[-1][:, :] - new_planet_pos[-1][0, :]
                new_sat_pos[-1] = new_sat_pos[-1][:, :] - new_planet_pos[-1][0, :]
            # Remove first values in lists that initialized them.
            new_planet_pos.pop(0)
            new_planet_vel.pop(0)
            new_sat_pos.pop(0)
            new_sat_vel.pop(0)
            new_sat_acc.pop(0)

        else:
            # For the number of items in the LSTM output sequence, run the
            # planet calcs for each loop.
            # Use initial loop to calculate the next time steps from initial
            # output of the neural net.
            new_planet_pos = [planet_pos]
            new_planet_vel = [planet_vel]
            new_sat_pos = [sat_pos]
            new_sat_vel = [sat_vel[-1]]
            new_sat_acc = [sat_acc[-1]]
            for i in range(0, num_out_seq_lstm):
                # Grab the accelerations acting on each body based on current
                # body positions.
                acceleration_np = self._future_calc_single_bod_acc_vectorized(
                    planet_pos=new_planet_pos[-1],
                    sat_pos=new_sat_pos[i], # Account for nn filling list
                    masses=masses
                )
                # Initialize the velocity matrix.
                # Only run calculations on planets
                velocity_np = new_planet_vel[-1]
                velocity_np = velocity_np.T.reshape(3, velocity_np.shape[0], 1) \
                              + (acceleration_np[:, 0:num_planets, :]
                                 * time_step)
                # Convert back to caching / tracking format and save to cache
                velocity_np = velocity_np.T.reshape(velocity_np.shape[1], 3)
                new_planet_vel.append(velocity_np[:num_planets, :])
                # Convert back and record acceleration history of sats.
                acceleration_np = \
                    acceleration_np.T.reshape(acceleration_np.shape[1], 3)
                new_sat_acc.append(acceleration_np[-num_sats:])
                # Calculate displacement and new location for planets
                # Displacement is based on the current velocity.
                velocities = new_planet_vel[-1]
                displacement_np = velocities * time_step
                # Update new positions of planets
                new_planet_pos.append(
                    new_planet_pos[-1] + displacement_np[:num_planets, :]
                )
                # If the initial loop, the run neural net inference.
                # Need vals from this to calculate acceleration on them at
                # each of the successive time steps.
                if i == 0:
                    # Use the given initialization values to run inference.
                    # Drop Z
                    sat_accels = np.swapaxes(sat_acc, 0, 1)[:, :, 0:2]
                    sat_masses = np.repeat(masses[-num_sats:],
                                           num_in_items_seq_lstm,
                                           axis=0)
                    sat_masses = sat_masses.reshape(
                        (num_sats, num_in_items_seq_lstm, 1)
                    )
                    sat_velocities = np.swapaxes(sat_vel, 0, 1)[:, :, 0:2]
                    # Create input and reshape to 3D for neural net.
                    input_sequence = np.concatenate(
                        [sat_masses, sat_accels, sat_velocities],
                        axis=2
                    )
                    predictions = neural_net.predict(input_sequence)
                    # Split predictions into displacement and
                    # pred_dis = predictions[:, :, :2]
                    # pred_vel = predictions[:, :, -2:]
                    zeroes = np.full(
                        (predictions.shape[0], predictions.shape[1], 1),
                        0.0,
                        dtype=np.float64
                    )
                    pred_dis = np.append(
                        predictions[:, :, :2],
                        zeroes,
                        axis=2
                    )
                    pred_vel = np.append(
                        predictions[:, :, -2:],
                        zeroes,
                        axis=2
                    )

                    # Reshape predictions to be by time step rather than
                    # by satellite.
                    pred_dis = np.swapaxes(pred_dis, 0, 1)
                    pred_vel = np.swapaxes(pred_vel, 0, 1)
                    # Update satellite positions and velocities using the
                    # predictions
                    for j in range(0, pred_dis.shape[0]):
                        # Add new positions to list using displacement.
                        new_sat_pos.append(
                            new_sat_pos[-1] + pred_dis[j]
                        )
                        # Add velocities for all satellites to list.
                        new_sat_vel.append(
                            pred_vel[j]
                        )
                # Set all positions relative to the sun
                new_planet_pos[-1] = \
                    new_planet_pos[-1][:, :] - new_planet_pos[-1][0, :]
                new_sat_pos[i + 1] = \
                    new_sat_pos[i + 1][:, :] - new_planet_pos[-1][0, :]
            # After for loop, remove initial values.
            new_planet_pos.pop(0)
            new_planet_vel.pop(0)
            new_sat_pos.pop(0)
            new_sat_vel.pop(0)
            new_sat_acc.pop(0)

        return new_planet_pos, new_planet_vel, new_sat_pos, new_sat_vel, new_sat_acc

    def _maintain_future_cache(self, output_queue, initial_planet_pos,
                               initial_planet_vel, initial_sat_pos,
                               initial_sat_vel, initial_sat_acc, masses,
                               time_step, nn_path, num_in_steps_lstm,
                               num_out_steps_lstm, keep_future_running,
                               ignore_nn):
        # Load neural net to run inference with.
        neural_net = tf.keras.models.load_model(nn_path)
        # Lists to cache calculations before they are pushed to the queue
        planet_pos_history = []
        planet_vel_history = []
        sat_pos_history = []
        sat_vel_history = []
        sat_acc_history = []

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Initialize all lists with first call to future with the
            # initial simulation values.
            future = executor.submit(
                self._future_compute_new_pos_vectorized,
                initial_planet_pos[-1],
                initial_planet_vel[-1],
                initial_sat_pos[-1],
                initial_sat_vel[-num_in_steps_lstm:],
                initial_sat_acc[-num_in_steps_lstm:],
                masses,
                time_step,
                neural_net,
                num_in_steps_lstm,
                num_out_steps_lstm,
                ignore_nn,
            )
            # Grab initial future and add to lists
            new_planet_pos, new_planet_vel, new_sat_pos, new_sat_vel, \
            new_sat_acc = future.result()
            del concurrent.futures.thread._threads_queues[
                list(executor._threads)[0]]
            # Add initialization to the lists
            planet_pos_history.extend(new_planet_pos)
            planet_vel_history.extend(new_planet_vel)
            sat_pos_history.extend(new_sat_pos)
            sat_vel_history.extend(new_sat_vel)
            sat_acc_history.extend(new_sat_acc)
            # Start another future
            future = executor.submit(
                self._future_compute_new_pos_vectorized,
                planet_pos_history[-1],
                planet_vel_history[-1],
                sat_pos_history[-1],
                np.array(sat_vel_history[-num_in_steps_lstm:]),
                np.array(sat_acc_history[-num_in_steps_lstm:]),
                masses,
                time_step,
                neural_net,
                num_in_steps_lstm,
                num_out_steps_lstm,
                ignore_nn
            )
            pre_q_max_size = 2000
            q_max_size = self._out_queue_max_size
            while keep_future_running.value == 1:
                if (len(planet_pos_history) <= pre_q_max_size) and future.done():
                    # Grab results from future and append to lists
                    new_planet_pos, new_planet_vel, new_sat_pos, new_sat_vel, \
                    new_sat_acc= future.result()
                    # Extend the current lists
                    planet_pos_history.extend(new_planet_pos)
                    planet_vel_history.extend(new_planet_vel)
                    sat_pos_history.extend(new_sat_pos)
                    sat_vel_history.extend(new_sat_vel)
                    sat_acc_history.extend(new_sat_acc)
                    # Start new future thread to compute more
                    future = executor.submit(
                        self._future_compute_new_pos_vectorized,
                        planet_pos_history[-1],
                        planet_vel_history[-1],
                        sat_pos_history[-1],
                        np.array(sat_vel_history[-num_in_steps_lstm:]),
                        np.array(sat_acc_history[-num_in_steps_lstm:]),
                        masses,
                        time_step,
                        neural_net,
                        num_in_steps_lstm,
                        num_out_steps_lstm,
                        ignore_nn
                    )
                # If the queue needs values, go and keep on pushing values.
                if planet_pos_history and (output_queue.qsize() < q_max_size):
                    output_list = [
                        planet_pos_history.pop(0),
                        planet_vel_history.pop(0),
                        sat_pos_history.pop(0),
                        sat_vel_history.pop(0),
                        sat_acc_history.pop(0)
                    ]
                    output_queue.put(output_list)
                # If the pre-q filled up, then just keep on trying to push
                # values to the queue.  Will pause here until queue has taken
                # more values.
                if (len(planet_pos_history) > pre_q_max_size):
                    time.sleep(1)
                    # output_list = [
                    #     planet_pos_history.pop(0),
                    #     planet_vel_history.pop(0),
                    #     sat_pos_history.pop(0),
                    #     sat_vel_history.pop(0),
                    #     sat_acc_history.pop(0)
                    # ]
                    # output_queue.put(output_list)'
            executor.shutdown(wait=False)
            print('Now outside the while loop')
        print('Now outside the with block')

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
        ignore_nn = False
        self._future_queue_process = Process(
            target=self._maintain_future_cache,
            args=(self._output_queue,
                  self._planet_pos_cache[0:self._len_lstm_in_seq],
                  self._planet_vel_cache[0:self._len_lstm_in_seq],
                  self._sat_pos_cache[0:self._len_lstm_in_seq],
                  self._sat_vel_cache[0:self._len_lstm_in_seq],
                  self._sat_acc_cache[0:self._len_lstm_in_seq],
                  self._masses,
                  self._time_step,
                  self._nn_path,
                  self._len_lstm_in_seq,
                  self._len_lstm_out_seq,
                  self._keep_future_running,
                  ignore_nn
                  )
        )
        self._future_queue_process.daemon = True
        self._future_queue_process.start()
        # Sleep until the queue is filled.
        while self._output_queue.qsize() < self._out_queue_max_size:
            time.sleep(3)

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
        # Grab the current working to use for referencing data files
        self._current_working_directory = \
            os.path.dirname(os.path.realpath(__file__))
        # Create neural network object that lets us run neural network
        # predictions as well.
        nn_name = 'my_model 99_95 8532.h5'
        self._nn_path = self._current_working_directory + "/nn/" + nn_name
        # Create neural net to use with future queue process
        # Since we are using an LSTM network, we will need to initialize the
        # the length of the sequence necessary for input into the LSTM.
        self._len_lstm_in_seq = 4
        self._len_lstm_out_seq = 10
        # Amount of time that has passed in a single time step in seconds.
        self._time_step = time_step
        # Grab info for creating background producer / consumer
        self._num_processes = cpu_count()
        # Create processing queues that producer / consumer will take
        # from and fill.
        self._out_queue_max_size = 500
        self._output_queue = Queue(self._out_queue_max_size)
        # Shared memory space to signal termination of threads when simulator's
        # destructor is called.
        self._keep_future_running = Value('I', 1)
        # Test list to append fake processed values to from the producer.
        self._test_output_list = []
        # Setup the initial set of bodies in the simulation by parsing from
        # config dataframe.
        self._parse_sim_config(in_config_df)

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

    def __del__(self):
        print("Destructor Called")
        self._keep_future_running.value = 0
        time.sleep(3)
        print('Now after sleep')
        self._future_queue_process.terminate()
        print('Process terminate called.')
        return

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
            # Keeping here for backup
            #self._compute_gravity_step_vectorized(ignore_nn=True)

            # Get next simulation state from the future queue and parse
            # out the various values from the list in the queue.
            next_state = self._output_queue.get()
            # Add the new state to all the caches.
            self._planet_pos_cache[self._curr_cache_index, :, :] = \
                next_state[0]
            self._planet_vel_cache[self._curr_cache_index, :, :] = \
                next_state[1]
            self._sat_pos_cache[self._curr_cache_index, :, :] = next_state[2]
            self._sat_vel_cache[self._curr_cache_index, :, :] = next_state[3]
            self._sat_acc_cache[self._curr_cache_index, :, :] = next_state[4]
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
            # Wait for future cache to recover from fast-forward.
            # while self._output_queue.qsize() < self._out_queue_max_size:
            #     time.sleep(3)
            while not self._output_queue.full():
                time.sleep(3)
        # If the time is between 0 and the max, set the current time step to 
        # the given time step.
        if (in_time_step >= self._len_lstm_in_seq) and \
                (in_time_step <= self._max_time_step_reached):
            # Update the simulator's time step
            self._current_time_step = in_time_step

