from pandas import read_excel
from scipy.sparse import dok_matrix
import numpy as np

def get_satellite_configs(filename):
    config = read_excel(filename, sheet_name=None)
    configs = []

    # separate by sheet (each sheet is one satellite)
    for satellite_page in config.values():
        satellite_config = {}
        satellite_config["name"] = satellite_page["Name"][0]
        satellite_config["mass"] = satellite_page["Mass"][0]
        satellite_config["alt"] = satellite_page["Altitude"][0]

        satellite_maneuvers_raw = []
        satellite_maneuvers_raw.append(satellite_page["MStart"])
        satellite_maneuvers_raw.append(satellite_page["DeltaVX"])
        satellite_maneuvers_raw.append(satellite_page["DeltaVY"])
        satellite_maneuvers_raw.append(satellite_page["DeltaVZ"])
        satellite_maneuvers_count = satellite_maneuvers_raw[0][len(satellite_maneuvers_raw[0]) - 1] + 1
        satellite_maneuvers = dok_matrix((satellite_maneuvers_count, 3), dtype=np.float32)

        for index, maneuver in enumerate(satellite_maneuvers_raw[0]):
            satellite_maneuvers[maneuver, 0] = satellite_maneuvers_raw[1][index]
            satellite_maneuvers[maneuver, 1] = satellite_maneuvers_raw[2][index]
            satellite_maneuvers[maneuver, 2] = satellite_maneuvers_raw[3][index]

        satellite_config["mans"] = satellite_maneuvers

        configs.append(satellite_config)

    return configs