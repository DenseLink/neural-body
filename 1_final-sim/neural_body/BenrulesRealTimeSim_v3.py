"""Real-Time basic simulator for planetary motions with neural network
inference for prediction of an arbitrary number of satellites.

This module contains the BenrulesRealTimeSim class, which creates a real time
simulator of the sun, planets, pluto, and an arbitrary number of satellites.
The initial simulation was forked from benrules2 on Github at the below link.
https://gist.github.com/benrules2/220d56ea6fe9a85a4d762128b11adfba
This simulator simulated a system for a fixed number of time steps and output
the results to a custom class and dictionary.

v2 of the simulator converted it to a real time simulator using a Pandas
dataframe to track simulation history and rewind back in time.  It also used
a feed-forward neural network that predicted the location of a specifically
trained body given the positions of every other body.  It also used
dictionaries to store current simulation states and calculated steps from
acceleration to velocity to displacement using loops.  Overall, this presented
the below challenges:
1. High memory usage.  Since the simulation for all time was stored to a
   Pandas Dataframe in memory, this meant that as the simulation ran, the
   memory usage would continually grow.
2. Slow simulation computations.  Many of the calculations used double and
   triple nested loops to calculte gravitational influence on a body and
   convert from the acceleration to velocity and velocity to displacement
   domains.
3. CPU idle time.  Since all calculations were done at run time, the simulator
   would sit at idle rather than continuing to perform calculations in the
   background.

v3, while taking inspiration from the original benrules simulator,
no longer resembles the original.  v3 is extensible to an arbitrary number of
satellites in the simulation.  The LSTM neural net it uses relies on the body
acceleration and current velocities to predict where the body will go for the
next 20 time steps.  The current behavior of the neural net is strange, which
is why in the config files, there is an option to turn it off and rely on the
simulator only.  Much more feature engineering, data generation, and
hyperparameter tuning is needed to accurately mimic orbital behavior.
One of the main challenges that arose as well with the LSTM network is slower
inference time.  Without background processing and predicting multiple time
steps at each inference, performance would be sub 3 fps.  Because of this
challenge and the v2 challenges, the below changes were made to v3.
1. Instead of using a Pandas dataframe that could require large amounts of
   memory with longer simulation durations, a new cache-archive system was
   developed to exchange simulation tracking between a numpy array that stores
   a sequence of 100 values in memory and an hdf5 file that stores a total
   record of the simulation so far.  As time jumps are performed, trackers and
   functions handle the flushing of new values from the cache to the archive
   and loading of time steps from the archive.  Having a history stored in
   numpy arrays also helps with vectorized computation since time steps don't
   need to be copied to other data structures for use in calculations.
2. Slow simulation computations were addressed by fully vectorizing all
   simulation computations using numpy linear algebra and getting rid of all
   computation loops.  This vectorization was done while maintaining the
   capability to add an arbitrary number of bodies to the simulation.
   Slow inference time with the neural network was addressed by trying to
   predict 20 time steps ahead given a sequence of 4 time steps as input to
   an LSTM network.  This enabled 1 inference cycle to produce 20 time steps.
3. Since the simulator is inherently a serial problem (in order to predict
   the next state, we must know the previous stat), the best option to address
   performance was to ensure a producer/consumer model was uses to
   continually run a producer in a background process that calculated future
   time steps even while the simulation is paused or when there is available
   CPU time on another processor.  Main simulation calculations were moved to
   an external process that calculates future time steps.  This process
   maintains a pre-queue of about 5000 future time steps.  It feeds a queue
   of 100 time steps between the background process and the main simulator that
   provides positions to the front end.  In order to reliably keep the main
   queue filled, the background process continually tries to keep the main
   queue filled with values from the pre-queue.  It is able to continually
   perform this task since the calculations filling the pre-queue are launched
   and performed in a separate thread.  This help mitigate the
   degradation of the buffer that the main queue provides.

In addition to the above improvements, benchmarking and queue monitoring occur
to ensure that the system runs at a steady pace that the simulator can keep
up with.  The user can burst to 2X or 1X, but if the queue begins to degrade,
the simulator will automatically revert back to 1X.  This is mainly for when
the neural network is used.  When not using the neural network, the simulator
will usually revert to a max framerate of 50 fps and even when running at
faster speeds, keep up without throttling back.

"""

# Imports
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
import h5py # Used for archive
from collections import deque # Used for burn maneuvers
# Imports for multiprocessing producer/consumer data model.
from multiprocessing import Process, Queue, Lock, cpu_count, Value
from threading import Thread

# Check if DISPLAY has been detected.  If not, assume WSL with pycharm and grab
# display connection.  Needed for pynput import.
# Only needed for launching from pycharm.
try:
    print(os.environ["DISPLAY"])
except KeyError as error:
    # If at this point, DISPLAY doesn't exist and needs to be set.
    line_list = []
    with open('/etc/resolv.conf') as f:
        for line in f:
            pass
        last_line = line
        line_list = last_line.split(' ')
        f.close()
    # Set the display
    os.environ["DISPLAY"] = line_list[-1].rstrip() + ":0"

# Need to run above before importing pynput
from pynput import keyboard
# Tkinter for progress bar
from tkinter import *
from tkinter import ttk


class BenrulesRealTimeSim:
    """
    Class containing real-time simulator for planet motions an an arbitrary
    number of satellites.  Can operate with and without a neural net to
    predict the motions of the satellites.

    Instance Variables:
    :ivar lock: Shared lock for stdout when troubleshooting or printing
        messages from background processes.
    :ivar time_step: Duration of time step the simulator is running on.
    :ivar current_working_directory: Current location of simuator script.
    :ivar nn_path: Path to the .h5 file containing the LSTM neural net.
    :ivar len_lstm_in_seq: Number of time steps in the input sequence to the
        LSTM network.
    :ivar len_lstm_out_seq: Number of time steps that can be predicted by the
        LSTM network in a single inference.
    :ivar num_processes: Max number of processes the CPU can handle.
    :ivar out_queue_max_size: Max size of the queue carrying information from
        the background simulation process and the functions that serve position
        data to the front end.
    :ivar output_queue: The queue that receives data from the background
        simulation process as the producer and the functions that serve
        position data to the front end as consumers.
    :ivar keep_future_running: Shared state integer that signals the while
        loop that runs the background simulation to stop, close out all threads
        and flush all queues.
    :ivar in_planet_df: Dataframe containing the initial planet data to launch
        the simulator.
    :ivar sim_archive_location: Location to store the .hdf5 file that contains
        all the cache data from the simulator for the entire simulation.
    :ivar latest_ts_in_archive: The latest time step stored currently in the
        hdf5 archive file.
    :ivar current_time_step: The current time step required to be served by the
        simulator.  Controls whether to grab data from archive, queue, or
        the future queue.
    :ivar max_time_step_reached: The maximum time step the simulator has ever
        served to the front end.
    :ivar curr_cache_size: Current size of the in-memory caches.
    :ivar max_cache_size: Max allowed size of the in-memory caches.
    :ivar latest_ts_in_cache: Latest time step stored in the current cache
        state.
    :ivar num_planets: Number of planets being simulated.
    :ivar num_sats: Number of satellites loaded into the simualtion.
    :ivar planet_pos_cache: Numpy array that caches the positions of all
        planets for the max size of the cache.
    :ivar planet_vel_cache: Numpy array that caches the velocities of all
        planets for the max size of the cache.
    :ivar sat_pos_cache: Numpy array that caches the positions of all
        satellites for the max size of the cache.
    :ivar sat_vel_cache: Numpy array that caches the velocities of all
        satellites for the max size of the cache.
    :ivar sat_acc_cache: Numpy array that caches the accelerations of all
        satellites for the max size of the cache.
    :ivar masses: Numpy array containing the masses of all bodies in the
        system.
    :ivar body_names: A list containing the names of all bodies in the system.
    :ivar future_queue_process: Object containing the background process that
        is continually calculating future time steps.
    :ivar max_fps: After simulation initialization, contains the maximum
        framerate the current system running the simulation can run at.
    """

    @staticmethod
    def _future_calc_single_bod_acc_vectorized(planet_pos,
                                               sat_pos,
                                               masses,
                                               burn_vec):
        """
        This is the function that calculates the acceleration on every every
        body due to the gravitational influence of every other body.

        :param planet_pos: Numpy array containing the positions of all planets.
        :param sat_pos: Numpy array containing the positions of all satellites.
        :param masses: Masses of all bodies in the simualtion.
        :param burn_vec: Numpy array containing all the burn maneuvers for
            the time steps currently being calculated.

        :return: Numpy array containing the calculated acceleration vectors.
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
        # Add burn maneuvers to satellites at end of acceleration matrix.
        temp_burn_vec = burn_vec.T.reshape(3, burn_vec.shape[0], 1)
        acceleration_np[:, -sat_pos.shape[0]:, :] = \
            acceleration_np[:, -sat_pos.shape[0]:, :] + temp_burn_vec
        return acceleration_np

    def _future_compute_new_pos_vectorized(self, planet_pos, planet_vel,
                                           sat_pos, sat_vel, sat_acc, masses,
                                           time_step, neural_net,
                                           num_in_items_seq_lstm,
                                           num_out_seq_lstm,
                                           result_queue, burn_vec,
                                           ignore_nn=False):
        """
        Function to calculate the positions of bodies in future time steps.

        The function takes in simulation data and calculates a number of
        future time steps equal to the number of time steps the neural network
        can output.  This function is meant to run as a separate thread and
        the results are placed in a result queue for later retrieval by the
        background simulation process.

        :param planet_pos: Numpy array containing the positions of all planets
            in the simulation.
        :param planet_vel: Numpy array containing the velocities of all planets
            in the simulation.
        :param sat_pos: Numpy array containing the positions of all satellites
            in the simulation.
        :param sat_vel: Numpy array containing the velocities of all satellites
            in the simulation.
        :param sat_acc: Numpy array containing the accelerations of all
            satellites in the system.
        :param masses: Numpy array containing the masses of all bodies in the
            system.
        :param time_step: Integer containing the duration of time step the
            simulation is running at in seconds.
        :param neural_net: Variable that points to the Tensorflow model object
            running in the background simulation process.
        :param num_in_items_seq_lstm: Number of time steps needed for input
            to the LSTM neural network.
        :param num_out_seq_lstm: Number of time steps in the output of the
            LSTM neural network.
        :param result_queue: Queue that carries result from computation thread
            to the background simulation process.
        :param burn_vec: Numpy array constructed with any additional
            accelerations due to burn maneuvers.
        :param ignore_nn: Boolean variable to bypass the neural network and
            use physics calculations only.

        :return: Results of function are placed in output queue and retrieved
            by background simulation process at a later point.
        """

        # Get the number of planets and satellites
        num_planets = planet_pos.shape[0]
        num_sats = sat_pos.shape[0]

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
                # Also use i to index the burn vector
                acceleration_np = self._future_calc_single_bod_acc_vectorized(
                    planet_pos=new_planet_pos[-1],
                    sat_pos=new_sat_pos[-1],
                    masses=masses,
                    burn_vec=burn_vec[i]
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
                # Check if burn maneuver occurred.  If it did, then set
                # z value to something other than 0
                if np.count_nonzero(burn_vec[i]) > 0:
                    new_sat_pos[i + 1][:, -1] = np.count_nonzero(burn_vec[i],
                                                                 axis=1)
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
                    masses=masses,
                    burn_vec=burn_vec[i]
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
                # Check if burn maneuver occurred.  If it did, then set
                # z value to something other than 0
                if np.count_nonzero(burn_vec[i]) > 0:
                    new_sat_pos[i+1][:, -1] = np.count_nonzero(burn_vec[i],
                                                              axis=1)
            # After for loop, remove initial values.
            new_planet_pos.pop(0)
            new_planet_vel.pop(0)
            new_sat_pos.pop(0)
            new_sat_vel.pop(0)
            new_sat_acc.pop(0)
        #Add reults to the result queue
        result_list = [new_planet_pos, new_planet_vel, new_sat_pos,
                       new_sat_vel, new_sat_acc]
        result_queue.put(result_list)

    @staticmethod
    def _create_acc_burn_vec(curr_time_step, num_out_steps_lstm,
                             sat_maneuver_queues):
        """
        Function to create a vector of time steps where each item in the
        vector is a matrix of satellites and additional burn values to use
        during the single body acceleration calculations.

        :param curr_time_step: Current simulation time step calculating the
            future from.
        :param num_out_steps_lstm: Number of time steps the LSTM network can
            output in a single inference.
        :param sat_maneuver_queues: List of queues containing the burn
            burn maneuvers for each satellite in the simulation.

        :return: Numpy array containing the time steps beign computed with
            burn maneuvers for each satellite.
        """
        # Calculate range of time steps the future thread will compute.
        curr_ts_range = range(
            curr_time_step,
            curr_time_step + num_out_steps_lstm
        )
        # Create 3D burn array that has dimensions (ts, sats, 3)
        burn_vec = np.full(
            (len(curr_ts_range), len(sat_maneuver_queues), 3),
            fill_value=np.float64(0),
            dtype=np.float64
        )
        # Loop over all time steps in the range and figure out if there are
        # any burns for each time step.
        for ts_idx, ts in enumerate(curr_ts_range):
            # Loop over each satellite and change burn_vec as needed from 0.0
            for sat_idx in range(0,len(sat_maneuver_queues)):
                # Make sure queue is not empty.
                if len(sat_maneuver_queues[sat_idx]) > 0:
                    # If the current time step has a maneuver for this sat, add
                    # it to the burn vector
                    if sat_maneuver_queues[sat_idx][0][0] == ts:
                        burn_vec[ts_idx, sat_idx, :] = \
                            sat_maneuver_queues[sat_idx].popleft()[1]
        return burn_vec

    def _maintain_future_cache(self, output_queue, initial_planet_pos,
                               initial_planet_vel, initial_sat_pos,
                               initial_sat_vel, initial_sat_acc, masses,
                               time_step, nn_path, num_in_steps_lstm,
                               num_out_steps_lstm, keep_future_running,
                               sat_maneuver_queues, ignore_nn):
        """
        :param output_queue: Queue that carries future time steps from this
            background simulation process to the part of the simulator
            serving body positions to the front end.
        :param initial_planet_pos: Numpy array containing the initial positions
            of the planets in the simulation.
        :param initial_planet_vel: Numpy array containing the initial
            velocities of the planets in the simulation.
        :param initial_sat_pos: Numpy array containing the initial positions of
            all satellites in the simulation.
        :param initial_sat_vel: Numpy array containing the initial velocities
            of all satellites in the simulation.
        :param initial_sat_acc: Numpy array containing the initial
            accelerations for all satellites in the simulation.
        :param masses: Numpy array containing all masses in the system.
        :param time_step: Time duration the simulation is performing
            calculations with.
        :param nn_path: File path of the .h5 file containing the trained
            LSTM neural network.
        :param num_in_steps_lstm: Number of time steps needed for input to
            the LSTM neural network.
        :param num_out_steps_lstm: Number of time steps output by the LSTM
            neural network.
        :param keep_future_running: Shared integer that signals the background
            process while loop to halt and kill all threads.
        :param sat_maneuver_queues: List of deques that have burn maneuvers
            for all satellites.
        :param ignore_nn: Boolean value of whether or not to ignore the
            neural network for calculating satellite positions.
        """

        # Local variable to keep track of the current simulation time step.
        # Assumes simulator initialized with enough time steps for the LSTM
        # network.
        curr_time_step = num_in_steps_lstm + 1
        # Load neural net to run inference with.
        neural_net = tf.keras.models.load_model(nn_path)
        # Lists to cache calculations before they are pushed to the output
        # queue which the simulator retrieves values from.
        planet_pos_history = []
        planet_vel_history = []
        sat_pos_history = []
        sat_vel_history = []
        sat_acc_history = []
        # Create another queue that the computing thread can add to.
        result_queue = Queue(maxsize=5)

        # Calculate burn maneuver vector to add to acceleration calcs.
        burn_vec = self._create_acc_burn_vec(curr_time_step,
                                             num_out_steps_lstm,
                                             sat_maneuver_queues)
        # Start a thread to initialize the lists.
        future = Thread(
            target=self._future_compute_new_pos_vectorized,
            args=(
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
                result_queue,
                burn_vec,
                ignore_nn
            ),
            daemon=True
        )
        future.start()
        # Wait for initial thread to complete.
        future.join()
        # Get results from the result queue
        results = result_queue.get()
        # Add initialization to the lists
        planet_pos_history.extend(results[0])
        planet_vel_history.extend(results[1])
        sat_pos_history.extend(results[2])
        sat_vel_history.extend(results[3])
        sat_acc_history.extend(results[4])
        # Increment time step
        curr_time_step = curr_time_step + num_out_steps_lstm
        # Start another thread and run the main process loop
        # Calculate burn maneuver vector to add to acceleration calcs.
        burn_vec = self._create_acc_burn_vec(curr_time_step,
                                             num_out_steps_lstm,
                                             sat_maneuver_queues)
        future = Thread(
            target=self._future_compute_new_pos_vectorized,
            args=(
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
                result_queue,
                burn_vec,
                ignore_nn
            ),
            daemon=True
        )
        future.start()
        # Run main program loop that generates threads
        pre_q_max_size = 5000
        q_max_size = self._out_queue_max_size
        while keep_future_running.value == 1:
            if keep_future_running == 0:
                break
            if (len(planet_pos_history) <= pre_q_max_size) and (not future.is_alive()):
                # Make sure the thread completes execution
                future.join()
                # Get results from the result queue
                results = result_queue.get()
                # Add results to the history lists
                planet_pos_history.extend(results[0])
                planet_vel_history.extend(results[1])
                sat_pos_history.extend(results[2])
                sat_vel_history.extend(results[3])
                sat_acc_history.extend(results[4])
                # Increment time step counter
                curr_time_step = curr_time_step + num_out_steps_lstm
                # Calculate burn maneuver vector to add to acceleration calcs.
                burn_vec = self._create_acc_burn_vec(curr_time_step,
                                                     num_out_steps_lstm,
                                                     sat_maneuver_queues)
                # Start new future thread to compute more
                future = Thread(
                    target=self._future_compute_new_pos_vectorized,
                    args=(
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
                        result_queue,
                        burn_vec,
                        ignore_nn
                    ),
                    daemon=True
                )
                future.start()
            # If the queue needs values, go and keep on pushing values.
            if planet_pos_history and (output_queue.qsize() < q_max_size):
                # Be careful where we are blocking.  Make sure we will
                # not be waiting to put item in queue
                if not output_queue.full():
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
            if (len(planet_pos_history) > pre_q_max_size + 500):
                time.sleep(1)
        # Flush the entire result queue so background threads can terminate.
        try:
            while not result_queue.empty():
                temp = result_queue.get_nowait()
        except:
            print("Exception when trying to flush result_queue")
            pass
        # If at this point, main loop broke.  Join rest of threads.
        future.join()
        return

    def _parse_sim_config(self, sat_config_file=None, planet_config_df=None):
        """
        Function to convert the given and system config files to appropriate
        data structures for use by the simulator.

        This function also launches the background simulation process and
        benchmarks the system's max, steady state, framerate.

        :param sat_config_file: Filename of config file containing satellite
            parameters.
        :param planet_config_df: Dataframe containing the simulation planet
            configuration.
        """

        # Using iterrows() to go over each row in dataframe and extract info
        # from each row.
        #self._bodies = []
        read_planet_pos = []
        read_planet_vel = []
        read_sat_pos = []
        read_sat_vel = []
        read_planet_masses = []
        read_sat_masses = []
        read_planet_names = []
        read_sat_names = []
        # Create list of queues to store the maneuvers for each satellite.
        sat_maneuver_queues = []

        # Read in the planets and their configurations.
        # Load planet configs
        for index, row in planet_config_df.iterrows():
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

        # Get the position data for earth and use to set the positions of the
        # satellites with altitudes above earth.
        try:
            earth_idx = read_planet_names.index('earth')
        except:
            print('Could not find earth in planet config file.')
        # Only using x and y for now for easier transformations.
        earth_loc = read_planet_pos[earth_idx][0:2]
        # Calculate unit vector for earth's location.
        earth_loc_unit = earth_loc / np.linalg.norm(earth_loc)

        # Load satellite and sim config
        ignore_nn = True
        #sat_config_location = self._current_working_directory + "/"
        config = pd.read_excel(sat_config_file, sheet_name=None)

        for page_idx, satellite_page in enumerate(config.values()):
            if page_idx == 0:
                nn_response = str(satellite_page["UseNeuralNet"][0])
                if (nn_response == 'yes') or (nn_response == "Yes"):
                    ignore_nn = False
                continue
            # Load initial values
            read_sat_names.append(
               str(satellite_page["Name"][0])
            )
            read_sat_masses.append(
               np.array([np.float64(satellite_page["Mass"][0])])
            )
            # Calculate absolute position of satellite
            temp_alt = int(satellite_page["Altitude"][0])
            rad_earth = 6371000
            temp_alt = temp_alt + rad_earth
            alt_vec = temp_alt * earth_loc_unit
            sat_pos = earth_loc + alt_vec
            read_sat_pos.append(
                [np.float64(sat_pos[0]),
                 np.float64(sat_pos[1]),
                 np.float64(0)]
            )
            # Calculate the satellite velocity perpendicular to location
            # vector going clockwise.
            trans_mat = np.array([[0, 1],[-1, 0]])
            vel_unit = trans_mat @ earth_loc_unit
            temp_speed = float(satellite_page["StartSpeed"][0])
            temp_vel = temp_speed * vel_unit
            # Add to velocity list and add z dimension back in.
            read_sat_vel.append(
               np.array([np.float64(temp_vel[0]),
                         np.float64(temp_vel[1]),
                         np.float64(0)])
            )
            # Populate satellite manuever list
            sat_mans = deque()
            for index, maneuver in enumerate(satellite_page["MStart"]):
                if not math.isnan(maneuver):
                    temp_man = [
                       int(maneuver),
                       np.array(
                           [np.float64(satellite_page["DeltaVX"][index]),
                            np.float64(satellite_page["DeltaVY"][index]),
                            np.float64(satellite_page["DeltaVZ"][index])]
                       )
                    ]
                    sat_mans.append(temp_man)
            sat_maneuver_queues.append(sat_mans)

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
                  sat_maneuver_queues,
                  ignore_nn
                  )
        )
        self._future_queue_process.daemon = True
        self._future_queue_process.start()
        # Time how long it takes to fill the queue and set an appropriate
        # framerate so the queue doesn't get drained too quick.
        start = time.time()
        # Create progress bar for simulation initialization.
        # Create tkinter window to show progress.
        root = Tk()
        root.title('Simulator Processing')
        screen_height = root.winfo_screenheight()
        screen_width = root.winfo_screenwidth()
        window_height = 50
        window_width = 1000
        geometry = "%sx%s+%s+%s" % (window_width,
                                    window_height,
                                    int(screen_width / 2 - window_width / 2),
                                    int(2))
        root.geometry(geometry)
        root.overrideredirect(True)
        root.resizable(width=FALSE, height=FALSE)
        # Create ttk progress bar widget.
        my_progress = ttk.Progressbar(root,
                                      orient=HORIZONTAL,
                                      length=window_width - 40,
                                      mode='determinate')
        my_progress.pack(pady=20)
        # Sleep until the queue is filled.
        while self._output_queue.qsize() < self._out_queue_max_size:
            progress_val = int(
                (self._output_queue.qsize() /
                 self._out_queue_max_size) * 100
            )
            my_progress['value'] = progress_val
            root.update()
            time.sleep(0.1)
        duration = time.time() - start
        root.quit()
        root.withdraw()
        self._max_fps = self._out_queue_max_size / duration
        if self._max_fps > 50:
            self._max_fps = 50
        print(f'System max supported frame-rate is: {self._max_fps}')

    def __init__(self, sat_config_file):
        """
        Simulation initialization function.

        :param sat_config_file: File path to the config file that contains
            the simulation configuration, satellites, and burn maneuvers for
            the satellites.
        """
        # Lock to take control of stdout
        self._lock = Lock()
        # Set the base simulation time step duration
        # 1 hour = 3600 seconds
        # 3 hour = 10800 seconds
        # 6 hour = 21600 seconds
        # 12 hour = 43200 seconds
        self._time_step = 21600
        # Grab the current working to use for referencing data files
        self._current_working_directory = \
            os.path.dirname(os.path.realpath(__file__))
        # Create neural network object that lets us run neural network
        # predictions as well.
        nn_name = '200epochs_6hr-ts.h5'
        self._nn_path = self._current_working_directory + "/nn/" + nn_name
        # Create neural net to use with future queue process
        # Since we are using an LSTM network, we will need to initialize the
        # the length of the sequence necessary for input into the LSTM.
        self._len_lstm_in_seq = 4
        self._len_lstm_out_seq = 10
        # Grab info for creating background producer / consumer
        self._num_processes = cpu_count()
        # Create processing queues that producer / consumer will take
        # from and fill.
        self._out_queue_max_size = 100
        self._output_queue = Queue(self._out_queue_max_size)
        # Shared memory space to signal termination of threads when simulator's
        # destructor is called.
        self._keep_future_running = Value('I', 1)
        # Read in planet config file separate from the sat config.
        try:
            in_planet_df = pd.read_csv(
                self._current_working_directory + '/sim_configs/' +
                "planet_config.csv"
            )
        except FileNotFoundError as error:
            print('Unable to find the planet config file.')
        # Setup the initial set of bodies in the simulation by parsing from
        # config dataframe.
        self._parse_sim_config(planet_config_df=in_planet_df,
                               sat_config_file=sat_config_file)

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
        """
        Simulation deconstructure. Used to signal background process to stop
        while loop and halt all computation threads.  Also flushed output
        queue of values allowing background process to close.
        """
        # Signal the background process to stop looping.
        self._keep_future_running.value = 0
        # Flush the entire output queue so background process can terminate.
        try:
            while not self._output_queue.empty():
                temp = self._output_queue.get_nowait()
        except:
            pass
        # Make double sure background process has terminated.
        self._future_queue_process.terminate()
        return

    def _calc_single_bod_acc_vectorized(self):
        """
        This function calculates the acceleration resulting from the
        gravitational influence of every body on every other body in the
        simulation.

        Mainly used for initialization of the simulator.  Fully vectorized to
        improve performance.

        :param body_index:
        :return: Acceleration resulting from gravitation influence of every
            body on every other body.
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
        Function to calculate the velocities resulting from new time step's
        acceleration.

        Mainly used for initialization of the simualtor.

        :param ignore_nn: Boolean value to decide whether or not to use the
            neural network during simulation initialization.
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
        """
        Function to calculate displacement and update the positions of all
        bodies.

        Mainly used for simulation initialization.

        :param ignore_nn: Boolean value to decide whether or not to use the
            neural network during simulation initialization.
        """
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
        """
        Function to coordinate between the velocity and location vectorized
        functions.

        :param ignore_nn: Boolean value to decide whether or not to use the
            neural network during simulation initialization.
        """
        self._compute_velocity_vectorized(ignore_nn=ignore_nn)
        self._update_location_vectorized(ignore_nn=ignore_nn)

    def _flush_cache_to_archive(self):
        """
        Function that handles memory cache management.

        The function will flush the cache if there are any values in the cache
        that are not yet in the archive.  It will then prefill the cache with
        enough values to continuously use  with the neural net.  If there are
        available values in the archive, it will then fill the latter end
        of the cache with time steps from the archive.
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
        """
        Function that serves the positions of all bodies in the next time step
        based on the internally tracked time step.

        Based on the current time step, this function will either retrieve
        values from the future queue that has new, calculated time steps,
        retrieve values from the cache directly, or fill the cache
        with values from the archive file and retrieve from the new, filled
        cache.

        :returns:
            - Numpy array with positions of all bodies in the simulation.
            - Boolean variable telling the front end to slow down request
              rate to allow background simulation process to fill back up.
        """
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
            # Get next simulation state from the future queue and parse
            # out the various values from the list in the queue.
            next_state = self._output_queue.get()
            # Check if orbits are getting outrageously large.  If so, just set to
            # 10^25.  Numpy fancy indexing!!
            default_orbit = np.float64(10e25)
            next_state[2][next_state[2] > default_orbit] = default_orbit
            # Add the new state to all the caches.
            self._planet_pos_cache[self._curr_cache_index, :, :] = \
                next_state[0]
            self._planet_vel_cache[self._curr_cache_index, :, :] = \
                next_state[1]
            self._sat_pos_cache[self._curr_cache_index, :, :] = next_state[2]
            self._sat_vel_cache[self._curr_cache_index, :, :] = next_state[3]
            self._sat_acc_cache[self._curr_cache_index, :, :] = next_state[4]
            # Increase the current cache size.
            self._curr_cache_size += 1
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
                    self._latest_ts_in_cache + 1
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
        # If the output queue is starting to get dangerously low, signal front
        # end to slow down simulation speed.
        slow_down = False
        if self._output_queue.qsize() < 20:
            slow_down = True
        # Return numpy array with the positions of all bodies in the simulation
        return simulation_positions, slow_down

    @property
    def current_time_step(self):
        """
        Getter that retrieves the current time step the simulator is at.

        :return current_time_step: Current time step the simulator is at.
        """
        return self._current_time_step

    @property
    def time_step_duration(self):
        """
        Getter that retrieves the duration of the time step in seconds that
        the simulation uses to calculate new positions.

        :return time_step_duration: Duration of the simulation time step in
            seconds.
        """
        return self._time_step

    @property
    def max_fps(self):
        """
        Getter that returns the calculated max framerate the current system
        can run at safely while maintaining future time step calculations.

        :return max_fps: Max framerate the current simulation can run at
            steadily.
        """
        return self._max_fps

    @property
    def body_names(self):
        """
        Getter that returns a list of the names of all bodies in the
        simualtion.

        :return body_names: List with the names of all bodies in the system.
        """
        return self._body_names

    @staticmethod
    def _on_press(key):
        """
        Function that checks if escape key has been pressed during time jump.

        :param key: Key pressed from pynput.

        :return: True or false if ESC is pressed.
        """
        if key == keyboard.Key.esc:
            return False

    @current_time_step.setter
    def current_time_step(self, in_time_step):
        """
        Setter to change the current time step of the simulator.  Essentially
        rewinding the simulation back to a point in its history.

        If negative time entered, default to 0 time.  If time entered past the
        maximum time reached, the simulator will "fast-forward" to that time
        step.  If fast forward takes a long time, it can be canceled by
        pressing the escape key.
        """

        # Store the old time step in case we need to reset.
        old_ts = self._current_time_step

        # Make sure we can't go back before the big bang.
        # Need to keep at least enough time steps for the LSTM network.
        if in_time_step <= self._len_lstm_in_seq:
            in_time_step = self._len_lstm_in_seq + 1
        # If time goes beyond the max time the simulator has reached, advance
        # the simulator to that time.
        if in_time_step > self._max_time_step_reached:
            # Create tkinter window to show progress.
            root = Tk()
            root.title('Simulator Processing')
            screen_height = root.winfo_screenheight()
            screen_width = root.winfo_screenwidth()
            window_height = 50
            window_width = 1000
            geometry = "%sx%s+%s+%s" % (window_width,
                                        window_height,
                                        int(screen_width/2 - window_width/2),
                                        int(2))
            root.geometry(geometry)
            root.overrideredirect(True)
            root.resizable(width=FALSE, height=FALSE)
            # Create ttk progress bar widget.
            my_progress = ttk.Progressbar(root,
                                          orient=HORIZONTAL,
                                          length=window_width - 40,
                                          mode='determinate')
            my_progress.pack(pady=20)
            with keyboard.Listener(on_press=self._on_press) as listener:
                while self._max_time_step_reached < in_time_step:
                    progress_val = int(
                        ((self._max_time_step_reached +
                          self._output_queue.qsize()) /
                         (in_time_step + self._out_queue_max_size)) * 100
                    )
                    my_progress['value'] = progress_val
                    root.update()
                    sim_positions = self.get_next_sim_state_v2()
                    if not listener.running:
                        self._current_time_step = old_ts
                        sim_positions = self.get_next_sim_state_v2()
                        root.quit()
                        root.withdraw()
                        break
                while not self._output_queue.full():
                    progress_val = int(
                        ((self._max_time_step_reached +
                          self._output_queue.qsize()) /
                         (in_time_step + self._out_queue_max_size)) * 100
                    )
                    my_progress['value'] = progress_val
                    root.update()
                    if not listener.running:
                        self._current_time_step = old_ts
                        root.quit()
                        root.withdraw()
                        break
                    time.sleep(0.5)
            # Close progress
            root.quit()
            root.withdraw()
            return
        # If the time is between 0 and the max, set the current time step to
        # the given time step.
        if (in_time_step >= self._len_lstm_in_seq) and \
                (in_time_step <= self._max_time_step_reached):
            # Update the simulator's time step
            self._current_time_step = in_time_step

