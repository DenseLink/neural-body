# Neural Body Project

## Overview
The Neural Body project combines the principles of physics with the power of neural networks to simulate and visualize gravitational interactions in a celestial system. This initiative leverages TensorFlow to predict the trajectories of planetary bodies, offering a unique approach to understanding and visualizing the dynamics of gravity.

## Components
The project is structured into several key components:

- **Model Deployments**: Scripts and pre-trained models for simulating gravitational orbits and predicting celestial body positions using neural networks.
- **Simulator**: A Python-based gravitational simulator that models the interaction between multiple celestial bodies.
- **Development Datasets**: Storage for datasets and notebooks related to neural network training and development.

### Key Files
- `Grav.py`: Demonstrates gravitational orbits using Pygame for visualization.
- `nn_model_loader.py`: Loads TensorFlow/Keras models for predicting planetary positions.
- `test_driver.py`: Tests the neural network model with actual data to predict positions.
- `Gravity.py`: Simulates gravitational forces and updates positions of celestial bodies.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Pygame
- Matplotlib
- Pandas
- Numpy

### Installation
Clone the repository to your local machine:
```
git clone https://github.com/DenseLink/neural-body.git
```

### Navigate to the project directory:
```
cd neural-body
```
Install the required Python packages:
```
pip install tensorflow pygame matplotlib pandas numpy
```
### Running the Simulations
To run the gravitational orbit simulation:
```
python neural_network_dev/model_deployments/0_FirstAttempt/Grav.py
```
To test the neural network model:
```
python neural_network_dev/model_deployments/0_FirstAttempt/test_driver.py
```
### Contributing
Contributions to enhance the Neural Body project are welcome. Feel free to fork the repository, make your changes, and submit a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for more details.

Acknowledgments
Special thanks to the TensorFlow and Pygame communities for their invaluable resources.
Inspired by the fascinating dynamics of celestial bodies and the potential of neural networks to simulate complex systems.
