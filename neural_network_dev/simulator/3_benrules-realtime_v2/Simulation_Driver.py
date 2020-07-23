"""
Script to call the real time simulation class and get
more data from it.

"""

# Imports
import sys
import pandas
from BenrulesRealTimeSim_v2 import BenrulesRealTimeSim

# Main code

def main():
    # Set time step (most likely in seconds) in simulation time.
    # How long the simulation steps forward at each iteration.
    time_step = 800
    pause_time_step = 250  # Pause the simulator at this time step and FF or RW
    # Total number of time steps performed by the simulation.
    # Total length of simulation = time_step * number_of_steps.
    number_of_steps = 300

    # Read simulator and satellite initial state from config .csv file.
    keep_trying_read = True
    config_file_location = "sim_configs/mars_sim_config.csv"
    sim_config_df = None
    while keep_trying_read:
        try:
            sim_config_df = pandas.read_csv(config_file_location,
                                            header=0)
            # If at this point, then file has been read.
            keep_trying_read = False
        except FileNotFoundError as error:
            print("Unable to find config file.  Try Again.")
            continue

    # Create simulator object
    simulation = BenrulesRealTimeSim(time_step=time_step,
                                     in_config_df=sim_config_df)

    # Run simulation
    while simulation.current_time_step < number_of_steps:
        print("Current Time Step: {}".format(simulation.current_time_step))
        # Pause and set a new time step for the simulator
        if (simulation.current_time_step == pause_time_step):
            new_ts = int(input("Enter a time step to jump to: "))
            simulation.current_time_step = new_ts
        # Get next state of the simulation.
        current_positions = simulation.get_next_sim_state_v2()
        #current_positions = simulation.get_next_sim_state()
        # Output the current position of all bodies
        print("-----------------------------------------------------------------")
        for key, coordinates in current_positions.items():
            print("Name: {} / Coordinates: {}".format(key, coordinates))
        print("-----------------------------------------------------------------")
    return None

if __name__ == "__main__":
    main()
    sys.exit()