"""
Description:
Quick driver program to test inputting data into the model class
and get a prediction out of it.

2020/06/14

"""

# Dunders

# Imports
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from nn_model_loader import NeuralNet

# Main classes and functions.


def main():
    """
    A quick program to load the class that loads the model to make predictions
    given an input vector of all other planet positions.  Input_data should be a
    numpy array.
    :return: None
    """
    # Load a simulation dataset and extract the system positioning data from
    # one of the rows.  Can later calculate error of the prediction.
    complete_motion_df = pd.read_csv("raw_model_output.csv")
    # Randomly select row to make prediction with and make numpy
    # arrays for input vector and actual simulation result.
    row_num = 445
    data_row = complete_motion_df.iloc[445]
    # Use last 3 columns of data row for keeping track of the simulation position
    # of the planet being predicted.
    model_pos = data_row.iloc[-3:].values
    # Create numpy array as input vector to model.
    input_data = data_row.iloc[0:-3].values
    # Reshape the numpy arrays for the input_data to an array of columns
    # rather than just an array.
    input_data = np.reshape(input_data, (-1, len(input_data)))
    # Create object to run predictions using the model specified by the path.
    network_location = "NN-Deploy-V1.01_2-layer_selu_lecun-normal_mae_Adam_lr-1e-6_bs-128_epoch-3500.h5"
    nn = NeuralNet(network_location, "Pluto")
    # Pass input_data to model to make a prediction.
    pred_pos = nn.make_prediction(input_data)

    #Try just straight loading the model and making predictions.
    #model = tf.keras.models.load_model(network_location)
    #pred_pos = model.predict(input_data)

    # Print model results and predicted results.
    # Calculate MAE and accuracy.
    print("Simulation Position: {}".format(model_pos))
    print("Predicted Position: {}".format(pred_pos))

    error_vector = 100 * (np.abs(model_pos - pred_pos) / (model_pos+0.00000001))
    print("Mean Absolute Percentage Error for all predictions: {}".format(error_vector))


if __name__ == "__main__":
    main()
    sys.exit()