"""
Class to load the below model spec and use it for prediction.

Layer Spec:
Model specification and layer spec:

_____________________________________________________________________________
# Use functional API to build basic NN architecture.
input_main = keras.layers.Input(shape=complete_motion_np.shape[1:])
hidden1 = keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal")(input_main)
hidden2 = keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal")(hidden1)
output_x = keras.layers.Dense(1, activation="linear", name="output_x")(hidden2)
output_y = keras.layers.Dense(1, activation="linear", name="output_y")(hidden2)
output_z = keras.layers.Dense(1, activation="linear", name="output_z")(hidden2)

model =  keras.Model(inputs=[input_main], outputs=[output_x, output_y, output_z])

#Set
input_losses = ["mae", "mae", "mae"]
input_loss_weights = [0.4, 0.4, 0.2]
input_optimizer = keras.optimizers.Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.999)
input_metrics = ["mae"]
input_num_epochs = 3500
input_batch_size = 128

model.summary()
_____________________________________________________________________________


"""

# Dunders

# Imports
import tensorflow as tf
import numpy as np


# Classes and Functions

class NeuralNet:
    """Class to load Tensorflow model stored in .h5 file and run
    inference with it. """

    # Instance Methods
    def __init__(self, model_path, planet_predicting):
        """
        Constructor for model class.  Loads the model into a private instance variable that
        can then be called on to make predictions on the position of Pluto.
        :param model_path: Path, including name, to the .h5 file storing the neural net.
        :param planet_predicting: Name of planet the model is predicting.  Just to keep track.
        :return: None.
        """
        self._model = tf.keras.models.load_model(model_path)
        self.planet_predicting = planet_predicting

    def make_prediction(self, input_vector):
        """
        Function to take a vector of all other planet positions and output the XYZ position
        of the planet we are predicting for the current time step.
        :param input_vector: Numpy array of all other planets and stars in the system.
        :return: Numpy array of X,Y,Z positions of planet we are predicting.
        """
        x_pred, y_pred, z_pred = self._model.predict(input_vector)
        # Process the predicted values to output a single numpy array rather
        # than three 2D arrays with a single value each.
        return np.array([x_pred[0, 0], y_pred[0, 0], z_pred[0, 0]])
