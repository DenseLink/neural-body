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

        def __init__(self, model_path, planet_predicting):
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
            self.planet_predicting = planet_predicting

        def make_prediction(self, input_vector):
            """
            Function to take a vector of all other planet positions and output
            the XYZ position of the planet being predicted for the current time
            step.

            :param input_vector: Numpy array of all other planets and stars
                in the system.
            :return: Dictionary of X,Y,Z positions of planet we are predicting.
            """

            x_pred, y_pred, z_pred = self._model.predict(input_vector)
            # Process the predicted values to output a single numpy array rather
            # than three 2D arrays with a single value each.
            return {self.planet_predicting: [x_pred[0, 0],
                                             y_pred[0, 0],
                                             z_pred[0, 0]]}

    # Class Variables

    # Planet data units: (location (m), mass (kg), velocity (m/s)

    # Dictionary containing the neural network file names.  Each neural network
    # is specially trained at predicting the position of that satellite in the
    # sol system.  Will expand neural network later to more situations.
    _neural_networks = {"mars":"MARS-Predict-NN-Deploy-V1.02-LargeDataset_"
                               "2-layer_selu_lecun-normal_mae_Adam_lr-1e-"
                               "5_bs-128_epoch-350.h5",
                       "pluto":"Predict-NN-Deploy-V1.02-LargeDataset_2-layer"
                               "_selu_lecun-normal_mae_Adam_lr-1e-6_bs-"
                               "128_epoch-250.h5"}

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

    def _parse_sim_config(self, in_df):
        """
        Function to convert Pandas dataframe containing simulator configuration
        to a list of Body objects that are digestible by the simulator.

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
        read_masses = []
        for index, row in in_df.iterrows():
            # Check if satellite or other.
            # If satellite, then set predicting name to choose the right
            # neural network.
            if row["satellite?"] == "yes":
                self._satellite_predicting_name = str(row["body_name"])
            self._bodies.append(
                self._Body(
                    location = self._Point(
                        float(row["location_x"]),
                        float(row["location_y"]),
                        float(row["location_z"])
                    ),
                    mass = float(row["body_mass"]),
                    velocity = self._Point(
                        float(row["velocity_x"]),
                        float(row["velocity_y"]),
                        float(row["velocity_z"])
                    ),
                    name = str(row["body_name"])
                )
            )
            # Also initialize the NEW numpy arrays for tracking current system
            # state
            read_planet_pos.append(
                np.array([np.float32(row["location_x"]),
                          np.float32(row["location_y"]),
                          np.float32(row["location_z"])])
            )
            read_planet_vel.append(
                np.array([np.float32(row["velocity_x"]),
                          np.float32(row["velocity_y"]),
                          np.float32(row["velocity_z"])])
            )
            #TODO: Add satellites in here.
            read_masses.append(
                np.array([np.float32(row["body_mass"])])
            )

        # Create numpy arrays to store body masses
        # Using 1 array for both planets and satellites.  Only calculation
        # using masses is the acceleration calculation and we don't need the
        # masses separated.
        self._num_sats = len(read_sat_pos)
        self._num_planets = len(read_planet_pos)
        self._num_bodies = self._num_sats + self._num_planets
        self._masses = np.stack(read_masses, axis=0)
        self._planet_pos = np.stack(read_planet_pos, axis=0)
        self._planet_vel = np.stack(read_planet_vel, axis=0)
        self._sat_pos = np.stack(read_sat_pos, axis=0)
        self._sat_vel = np.stack(read_sat_vel, axis=0)

        # self._masses = np.full(
        #     (self._num_bodies, 1),
        #     np.nan,
        #     dtype=np.float32
        # )
        # # Create arrays to keep track of the current system state.
        # self._planet_pos = np.full(
        #     (self._num_planets, 3),
        #     np.nan,
        #     dtype=np.float32
        # )
        # self._planet_vel = np.full(
        #     (self._num_planets, 3),
        #     np.nan,
        #     dtype=np.float32
        # )
        # self._sat_pos = np.full(
        #     (self._num_sats, 3),
        #     np.nan,
        #     dtype=np.float32
        # )
        # self._sat_vel = np.full(
        #     (self._num_sats, 3),
        #     np.nan,
        #     dtype=np.float32
        # )


    def __init__(self, in_config_df, time_step=100):
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

        # Setup the initial set of bodies in the simulation by parsing from
        # config dataframe.
        self._satellite_predicting_name = None
        self._bodies = None
        self._parse_sim_config(in_config_df)  #self._bodies initialized.

        # Setup pandas dataframe to keep track of simulation history.
        #
        # Pandas dataframe is easy to convert to any data file format
        # and has plotting shortcuts for easier end-of-simulation plotting.
        self._body_locations_hist = self._initialize_history()
        # Amount of time that has passed in a single time step in seconds.
        self._time_step = time_step
        # Grab the current working to use for referencing data files
        self._current_working_directory = \
            os.path.dirname(os.path.realpath(__file__))
        # Create neural network object that lets us run neural network
        # predictions as well.
        # Default to mars model if key in dictionary not found.
        nn_path = self._current_working_directory + "/nn/" \
                  + self._neural_networks.get(
            str(self._satellite_predicting_name),
            "mars"
        )
        self._nn = self._NeuralNet(
            model_path=nn_path,
            planet_predicting=self._satellite_predicting_name
        )
        # Add current system state to the history tracking.
        coordinate_list = []
        for target_body in self._bodies:
            coordinate_list.append(target_body.location.x)
            coordinate_list.append(target_body.location.y)
            coordinate_list.append(target_body.location.z)
        # Store coordinates to dataframe tracking simulation history
        self._body_locations_hist.loc[len(self._body_locations_hist)] \
            = coordinate_list
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
        # Create Caches
        self._planet_pos_cache = np.full(
            (self._max_cache_size, self._num_planets, 3),
            np.nan,
            dtype=np.float32
        )
        self._planet_vel_cache = np.full(
            (self._max_cache_size, self._num_planets, 3),
            np.nan,
            dtype=np.float32
        )
        self._sat_pos_cache = np.full(
            (self._max_cache_size, self._num_sats, 3),
            np.nan,
            dtype=np.float32
        )
        self._sat_vel_cache = np.full(
            (self._max_cache_size, self._num_sats, 3),
            np.nan,
            dtype=np.float32
        )
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
            sat_group.create_dataset("loc_archive",
                                     (0, self._num_sats, 3),
                                     maxshape=(None, self._num_planets, 3))
        # Keep track of the latest time step stored in the archive.  Will be
        # used to determine if data from cache actually need flushing.
        self._latest_ts_in_archive = 0

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
            # Update the NEW numpy arrays that keep track of current
            # system state as well.
            self._planet_vel[body_index] = np.array([target_body.velocity.x,
                                                     target_body.velocity.y,
                                                     target_body.velocity.z])

    def _update_location(self):
        """
        Calculates next location of each body in the system.

        This method, assuming the _compute_velocity method was already called,
        takes the new velocities of all bodies and uses the defined time step
        to calculate the resulting displacement for each body over that time
        step.  The displacement is then added to the current positions in order
        to get the body's new location.
        """

        for body_index, target_body in enumerate(self._bodies):
            target_body.location.x += target_body.velocity.x * self._time_step
            target_body.location.y += target_body.velocity.y * self._time_step
            target_body.location.z += target_body.velocity.z * self._time_step
            # Update the NEW numpy arrays that keep track of current
            # system state as well.
            self._planet_pos[body_index] = np.array([target_body.location.x,
                                                     target_body.location.y,
                                                     target_body.location.z])

    def _compute_gravity_step(self):
        """
        Calls the _compute_velocity and _update_location methods in order to
        update the system state by one time step.
        """

        self._compute_velocity()
        self._update_location()

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
            # Calculate portion of cache that should be flushed
            if self._latest_ts_in_archive in range(beg_ts_in_cache,
                                                   self._latest_ts_in_cache):
                beg_cache_index = self._latest_ts_in_cache \
                                  - self._latest_ts_in_archive - 1
            else:
                # Flush the whole cache to the end of the archive
                beg_cache_index = 0
            end_cache_index = self._curr_cache_index
            cache_flush_size = end_cache_index - beg_cache_index + 1
            # Open archive for appending, resize datasets, and append current
            # caches to the end of their respective datasets.
            with h5py.File(self._sim_archive_loc, 'a') as f:
                # Grab pointer to the datasets
                planet_dset = f['planets/loc_archive']
                # sat_dset = f['satellites/loc_archive']
                # Resize the datasets to accept the new set of cache of data
                planet_dset.resize((planet_dset.shape[0] + cache_flush_size),
                                   axis=0)
                # sat_dset.resize((planet_dset.shape[0] + cache_flush_size),
                #                 axis=0)
                # Save data to the file
                planet_dset[-cache_flush_size:] = \
                    self._planet_pos_cache[beg_cache_index:end_cache_index + 1]
                # sat_dset[-self._curr_cache_size:] = \
                #     self._sat_pos_cache[0:self._curr_cache_size]
                # Update what is the latest time step stored in the archive.
                # Can assume it is the size of the archive.
                self._latest_ts_in_archive = planet_dset.shape[0]
        # TODO: When flushing cache, make sure to always include 1 past time step

        # Reset the caches whether data was flushed or not.
        self._latest_ts_in_cache = None
        self._curr_cache_size = 0
        self._curr_cache_index = -1

    def _update_caches(self, target_time_step):
        """
        Update the cache with necessary data from archive.
        Returns:

        """
        # Flush current cache of any data not yet in the archive.
        self._flush_cache_to_archive()
        # Figure out what data to grab from archive.  If the target_time_step
        # plus the max cache size is larger than the latest time step in the
        # archive, then we can partially fill the cache and set the cache size
        # accordingly.


    def get_next_sim_state_v2(self):
        # Push current time step counter and figure out how to get current
        # time step's value
        self._current_time_step += 1

        # We know we need to get positions through calculation and inference
        # rather than from cache when our current simulation has reached
        # the max time step.
        # If the current_time_step-1 == the max time step reached, then we know
        # we have gone beyond the current simulation and need to compute the
        # next simulation time step.
        if self._current_time_step-1 == self._max_time_step_reached:
            # Move forward the maximum time step the simulation has reached.
            self._max_time_step_reached += 1
            # Extract last row of dataframe recording simulator history, remove
            # the planet we are trying to predict from the columns, and convert
            # to numpy array as the input vector to the neural network.
            prediction_data_row = self._body_locations_hist.iloc[-1, :].copy()
            # Compute the next time step and update positions of all bodies
            # in self_bodies.
            self._compute_gravity_step()
            # Format position data for each planet into simple lists.
            # Dictionary key is the name of the planet.
            # Used to provide position data back to front end.
            simulation_positions = {}
            # Also create a coordinate list that can be added as row to the
            # history dataframe
            coordinate_list = []
            # Check if cache is full or within tolerance where we need to
            # flush cache to archive file
            if self._curr_cache_index == self._max_cache_size - 1:
                self._flush_cache_to_archive()
            # Increase current cache size.
            self._curr_cache_size += 1
            self._curr_cache_index += 1
            # Add current time step to the cache
            self._planet_pos_cache[self._curr_cache_index] = self._planet_pos
            self._planet_vel_cache[self._curr_cache_index] = self._planet_vel

            # Update the old cache
            for index, target_body in enumerate(self._bodies):
                # Add data to the old cache
                simulation_positions.update(
                    {target_body.name: [target_body.location.x,
                                        target_body.location.y,
                                        target_body.location.z]})
                coordinate_list.append(target_body.location.x)
                coordinate_list.append(target_body.location.y)
                coordinate_list.append(target_body.location.z)
                # if self._curr_cache_size <= self._max_cache_size:
                #     # Try adding to numpy location cache
                #     # Using size before incrementing by 1 to account for index
                #     # of np arrays starting at 0.
                #     self._planet_pos_cache[self._curr_cache_index, index, 0] =\
                #         target_body.location.x
                #     self._planet_pos_cache[self._curr_cache_index, index, 1] =\
                #         target_body.location.y
                #     self._planet_pos_cache[self._curr_cache_index, index, 2] =\
                #         target_body.location.z

            # Update the latest time step stored in the cache
            self._latest_ts_in_cache = self._current_time_step
            # Store coordinates to dataframe tracking simulation history
            self._body_locations_hist.loc[
                len(self._body_locations_hist)] = coordinate_list

        # If the current time step is less than the max time step reached and
        # the current time step is in the range of time steps in the cache,
        # then we can go ahead and grab position data from the cache.
        elif (self._current_time_step < self._max_time_step_reached) and \
                (self._current_time_step in range(
                    self._latest_ts_in_cache - self._curr_cache_size + 1,
                    self._latest_ts_in_cache + 1
                )):
            # If the current time step is in the range of time steps in the
            # cache, we can assume we can increase the cache index and grab
            # position data from that index.
            self._curr_cache_index += 1
            # Format position data for each planet into simple lists.
            # Dictionary key is the name of the planet.
            simulation_positions = {}
            # Run through each body and update the current body positions.
            for body_index, target_body in enumerate(self._bodies):
                target_body.location.x = \
                    self._planet_pos_cache[self._curr_cache_index,
                                           body_index, 0]
                target_body.location.y = \
                    self._planet_pos_cache[self._curr_cache_index,
                                           body_index, 1]
                target_body.location.y = \
                    self._planet_pos_cache[self._curr_cache_index,
                                           body_index, 2]
            # Assuming previous time step is always available in cache,
            # grab previous time step and use to run inference.
            prediction_data_row = pd.DataFrame(
                data=self._planet_pos_cache[self._current_time_step-1].reshape((-1,)),
                columns=list(self._body_locations_hist.columns)
            )

        # If the current time step is less than the max time step reached and
        # the current time step is NOT in the range of the cache, we need to
        # update the cache before proceeding with getting location information.
        elif (self._current_time_step < self._max_time_step_reached) and \
                (self._current_time_step not in range(
                    self._latest_ts_in_cache - self._curr_cache_size + 1,
                    self._latest_ts_in_cache + 1
                )):
            print("FUUCCKCKKCKCKCK")

        else:
            # Extract row of previous time step to current time step for
            # constructing input vector to neural network.
            prediction_data_row = self._body_locations_hist.iloc[
                                  self._current_time_step - 1, :].copy()
            coordinate_list = self._body_locations_hist.iloc[
                              self._current_time_step, :].tolist()
            # Format position data for each planet into simple lists.
            # Dictionary key is the name of the planet.
            simulation_positions = {}
            # Iterate over all columns in the extracted row and extract the
            # planet name along with the planet name.
            col_names = list(self._body_locations_hist.columns)
            index = 0
            while index < len(col_names):
                # Extract body name from columns
                body_name = col_names[index].split('_')[0]
                simulation_positions.update(
                    {body_name: [coordinate_list[index],
                                 coordinate_list[index + 1],
                                 coordinate_list[index + 2]]}
                )
                # Advance index by 3 columns to skip x, y, and z columns.
                index += 3
            # Push current time step forward 1
            self._current_time_step += 1

        # Predict planet location using neural network.
        # Need to use the name of the planet to find which one to extract from
        # the input vector.
        # Drop columns from dataframe for the planet we are trying to predict.
        prediction_data_row = prediction_data_row.drop(
            [self._satellite_predicting_name + "_x",
             self._satellite_predicting_name + "_y",
             self._satellite_predicting_name + "_z"]
        )
        input_vector = prediction_data_row.values.reshape(1, -1)
        # Predict position of satellite
        pred_pos = self._nn.make_prediction(input_vector)

        # Return dictionary with planet name as key and a list with each planet
        # name containing the coordinates
        return simulation_positions, pred_pos

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

        # Depending on the current time step and max time step reached, figure
        # out where to pull data from to make prediction with neural network
        # and how to create the next time step of the simulation.  If the
        # current time step is less than the max time step, then pull sim
        # data from the history dataframe.  If current time step is equal to
        # the max time step, then continue calculating positions with the
        # simulator.
        if self._current_time_step == self._max_time_step_reached:
            # Extract last row of dataframe recording simulator history, remove
            # the planet we are trying to predict from the columns, and convert
            # to numpy array as the input vector to the neural network.
            prediction_data_row = self._body_locations_hist.iloc[-1, :].copy()
            # Compute the next time step and update positions of all bodies
            # in self_bodies.
            self._compute_gravity_step()
            # Format position data for each planet into simple lists.
            # Dictionary key is the name of the planet.
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
            # Push time step counters forward
            self._current_time_step += 1
            self._max_time_step_reached += 1
        else:
            # Extract row of previous time step to current time step for
            # constructing input vector to neural network.
            prediction_data_row = self._body_locations_hist.iloc[
                self._current_time_step - 1, :].copy()
            coordinate_list = self._body_locations_hist.iloc[
                self._current_time_step, :].tolist()
            # Format position data for each planet into simple lists.
            # Dictionary key is the name of the planet.
            simulation_positions = {}
            # Iterate over all columns in the extracted row and extract the
            # planet name along with the planet name.
            col_names = list(self._body_locations_hist.columns)
            index = 0
            while index < len(col_names):
                # Extract body name from columns
                body_name = col_names[index].split('_')[0]
                simulation_positions.update(
                    {body_name: [coordinate_list[index],
                                 coordinate_list[index + 1],
                                 coordinate_list[index + 2]]}
                )
                # Advance index by 3 columns to skip x, y, and z columns.
                index += 3
            # Push current time step forward 1
            self._current_time_step += 1

        # Predict planet location using neural network.
        # Need to use the name of the planet to find which one to extract from
        # the input vector.
        # Drop columns from dataframe for the planet we are trying to predict.
        prediction_data_row = prediction_data_row.drop(
            [self._satellite_predicting_name + "_x",
             self._satellite_predicting_name + "_y",
             self._satellite_predicting_name + "_z"]
        )
        input_vector = prediction_data_row.values.reshape(1, -1)
        # Predict position of satellite
        pred_pos = self._nn.make_prediction(input_vector)

        # Return dictionary with planet name as key and a list with each planet
        # name containing the coordinates
        return simulation_positions, pred_pos

    @property
    def body_locations_hist(self):
        """
        Getter that returns a Pandas dataframe with the entire simulation
        history.

        :return body_locations_hist: Pandas dataframe containing the entire
            history of the simulation.  The positional data of all bodies
            over all time steps.
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
    def satellite_predicting_name(self):
        """
        Getter that retrieves the name of the planet the neural network is
        trying to predict the position of.

        :return planet_predicting_name:  Name of the planet the neural
            network is trying to predict.
        """
        return self._satellite_predicting_name

    @property
    def current_time_step(self):
        """
        Getter that retrieves the current time step the simulator is at.

        :return current_time_step: Current time step the simulator is at.
        """
        return self._current_time_step

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
        if in_time_step < 0:
            in_time_step = 0
        # If time goes beyond the max time the simulator has reached, advance
        # the simulator to that time.
        if in_time_step > self._max_time_step_reached:
            while self._max_time_step_reached < in_time_step:
                current_positions, predicted_position \
                    = self.get_next_sim_state_v2()
        # If the time is between 0 and the max, set the current time step to 
        # the given time step.
        if (in_time_step >= 0) and \
                (in_time_step <= self._max_time_step_reached):
            # Update the simulator's time step
            self._current_time_step = in_time_step
            # Flush the current cache of data not currently in the archive.
            self._flush_cache_to_archive()
            #TODO: Update cache with portion from archive.
            # Will update current cache index in this function.

    @property
    def max_time_step_reached(self):
        """
        Getter that retrieves the maximum time step the simulation has reached.

        :return max_time_step_reached: Max time step the simulation has
            reached.
        """
        return self._max_time_step_reached
