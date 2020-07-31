from benrules_v2_simmod import benrules_v2
import numpy as np
from tqdm import tqdm
import h5py
import sys
from random import randint


def main():
    # Setup simulation settings
    time_step = 720
    number_of_steps = 200000
    report_frequency = 5

    # Grab a random state of the universe to start from.
    # Read in the binary numpy files storing the acceleration,
    # velocity, position, and mass for
    # every body at every time step.
    universe_data_folder = \
        "computed_data/248yr_800ts-planets_only/"

    # Get time step to grab data from simulation.
    # 248 years (orbit of pluto) = 7826284800 seconds
    universe_sim_length: int = 7826284800
    universe_sim_time_step: int = 800
    universe_num_sim_time_steps: int = \
        universe_sim_length // universe_sim_time_step
    universe_time_step: int = randint(0, universe_num_sim_time_steps)
    # Attempting to open hdf5 file and save a portion of it to a numpy array.
    initial_pos = None
    initial_vel = None
    masses = None
    # Read in position data for that point in time.
    with h5py.File(universe_data_folder + 'p.hdf5', 'r') as f:
        data = f['pos']
        initial_pos = data[universe_time_step]
    # Read in velocity data for that point in time.
    with h5py.File(universe_data_folder + 'v.hdf5', 'r') as f:
        data = f['vel']
        initial_vel = data[universe_time_step]
    # Read in masses for the planets from the universe simulation.
    masses = np.load(universe_data_folder + 'm.npy')
    # Setup the initial set of planets for the simulation
    initial_bodies = []
    planet_name_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter',
                        'saturn', 'uranus', 'neptune', 'pluto']
    for idx, name in enumerate(planet_name_list):
        initial_bodies.append(
            benrules_v2.body(
                location=benrules_v2.point(
                    initial_pos[idx, 0],
                    initial_pos[idx, 1],
                    initial_pos[idx, 2]
                ),
                mass=masses[idx],
                velocity=benrules_v2.point(
                    initial_vel[idx, 0],
                    initial_vel[idx, 1],
                    initial_vel[idx, 2]
                ),
                name=name
            )
        )

    # ---> Randomly select a variety of satellites in the specified ranges.
    # About 200 kg to 4000 kg for small to large sats.
    sat_mass_range = (200, 4000)
    # Set range of initial speeds for satellites.
    # Escape speed from earth is about 11.2 km/s.  11,200 m/s
    # Avg speed of a satellite is 7600 m/s
    sat_speed_range = (15000, 20000)
    # Set range of orbits above earth for satellites.  Take into account
    # the Radius of earth since the mass will be located at its center.
    # Avg orbit is 160 to 2000 km.  160000 to 2000000 m
    rad_earth = 6371000
    sat_alt_range = (rad_earth + 1000000, rad_earth + 5000000)

    # ---> Create list of random satellites starting from earth.
    # Get location of earth.
    earth_spec = initial_bodies[3]
    sat_list = []
    num_sats_to_sim = 10
    for i in range(0, num_sats_to_sim):
        # Randomly select mass, speed, and altitude above earth for sat.
        temp_mass = randint(sat_mass_range[0], sat_mass_range[1])
        temp_speed = randint(sat_speed_range[0], sat_speed_range[1])
        temp_alt = randint(sat_alt_range[0], sat_alt_range[1])
        # Calculate location of sat from selected altitude.
        earth_loc = np.array([earth_spec.location.x,
                              earth_spec.location.y])
        earth_loc_unit = earth_loc / np.linalg.norm(earth_loc)
        alt_vec = temp_alt * earth_loc_unit
        sat_loc = earth_loc + alt_vec
        # Calculate the velocity perpendicular to the location vector going
        # clockwise.
        # Transformation matrix for rotating by 90 degrees
        trans_mat = np.array([[0, 1],[-1, 0]])
        vel_unit = trans_mat @ earth_loc_unit
        temp_vel = temp_speed * vel_unit
        # Create satellite
        sat_list.append(
            benrules_v2.body(
                location=benrules_v2.point(
                    sat_loc[0], sat_loc[1], 0
                ),
                mass=temp_mass,
                velocity=benrules_v2.point(
                    temp_vel[0], temp_vel[1], 0
                ),
                name='sat' + str(i)
            )
        )

    # Create list of lists for each simulation.  The sublists are the
    # satellite(s) to include in addition to the universe.
    # sim1 = [
    #     benrules_v2.body(
    #         location=benrules_v2.point(0, 149.602e9, 0),
    #         mass=4500,
    #         velocity=benrules_v2.point(7800, 0, 0)
    #     )
    # ]
    # sim2 = [
    #     benrules_v2.body(
    #         location=benrules_v2.point(0, 149.603e9, 0),
    #         mass=5000,
    #         velocity=benrules_v2.point(7800, 0, 0)
    #     )
    # ]
    # sim_list = [sim1, sim2]

    # Open hdf5 file for writing and appending.  Acts as cache to store each
    # simulation's data.
    # https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
    with h5py.File('sim_cache/sim_cache.hdf5', 'w') as f:
        # Loop over each list of satellite(s) and append the simulation
        # data into an HDF5 file
        for index, satellite in enumerate(sat_list):
            # Create an instance of the simulator to run with the current
            # satellite.
            current_sim = benrules_v2(
                time_step=time_step,
                number_of_steps=number_of_steps,
                report_frequency=report_frequency,
                bodies=initial_bodies,
                additional_bodies=[satellite]
            )
            current_sim.run_simulation()
            # Remove last element of each cache to account for displacement
            # missing from last value in cache.
            current_sim.acc_np = current_sim.acc_np[:-1].copy()
            current_sim.vel_np = current_sim.vel_np[:-1].copy()
            current_sim.pos_np = current_sim.pos_np[:-1].copy()
            current_sim.dis_np = current_sim.dis_np[:-1].copy()

            # Create group in hdf5 file for the current sim
            sim_group = f.create_group('sim_' + str(index))
            sim_group.create_dataset('acc', data=current_sim.acc_np)
            sim_group.create_dataset('vel', data=current_sim.vel_np)
            sim_group.create_dataset('pos', data=current_sim.pos_np)
            sim_group.create_dataset('dis', data=current_sim.dis_np)
            sim_group.create_dataset('mass', data=current_sim.mass_np)
            print("Sim {} of {} saved to cache.".format(
                index+1, len(sat_list)))

    print("Done running simulations.")
    # # For the designated simulation, visualize the positions of all bodies.
    # with h5py.File('sim_cache/sim_cache.hdf5', 'r') as f:
    #     # Open position data for the first simulation and load into memory.
    #     pos_data = f['sim_0/pos'][()]
    #     # Convert the data to a list of time series for plotting
    #     pos_x_list, pos_y_list = plot_data_conv_3D_np_pos_to_2D_pos_list(
    #         pos_data)
    #     # Plot the converted data.
    #     fig = plot_2D_body_time_series(
    #         pos_x_list=pos_x_list,
    #         pos_y_list=pos_y_list,
    #         plot_width=800,
    #         plot_height=800,
    #         title="Universe Paths 100,000 Time Steps in future"
    #     )
    #
    #     bokeh.plotting.show(fig)

if __name__ == "__main__":
    main()
    sys.exit()