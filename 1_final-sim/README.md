# Neural Body N-Body Simulator

## Summary
An n-body simulator powered by a neural network.

Neural Body is an n-body simulator is currently a demonstration of substituting 
calculations for planetary motion with a neural network.  The user can choose
from a selection of provided config files or modify config files with 
additional satellites and burn maneuvers.

Planets are calculated with physics while satellites use and LSTM neural 
network to predict what the satellite's next 20 time steps of displacement and 
velocity will look like given a sequence of 4 time steps. There is the option
to ignore the neural network in the config file.  The user might opt to do this
since LSTM neural network inference is computationally intensive.  Raw physics
calculations are vectorized and should be more performant.

Below is a Google Colab notebook that shows the output of a training run for the 
neural network that predicts Mars' position.  Code that generated the training 
data and performed preprocessing is not included.  Data file is also not included.
The link is purely to view code, learning curves, and results.

<a href="https://colab.research.google.com/drive/19-pUEmro6ajxLlUAPunM66i42gAaqrPz?usp=sharing" target="_blank"> LSTM Neural Network Training </a>
<br>

---
## Table of Contents
- Installation
- Usage
- Documentation
- Improvements Since Alpha
- Team
- License

[![Game Overview Image](https://raw.githubusercontent.com/nedgar76/neural-body/demo-sim/0_demo-sim/readme_resources/overview_screenshot.png?token=ALC2NMM5G56RZFD237TQX32677FSA)]()
---
## Installation
### Requirements
- Compatible debian-based Linux distro.  Ubuntu 20.04 Preferred.
- Python 3.8 or higher.
- PyGame 2.0.0.dev10 or higher.
- TensorFlow 2.2.0
- Pandas 1.0.5
- Numpy 1.19.0
All dependencies above except for Python 3.8 should install when `pip install` is run.

### Setup

Installing from .tar.gz:
- Download `neural_body-0.1.2.tar.gz` from the `dist` directory.
- Navigate to local folder where download is located.
- Use `pip install neural_body-0.1.2.tar.gz`
- Use the `neural_body` command to run the simulator from terminal.

Installing from PyPi:
- PyPi project page is located <a href="https://pypi.org/project/neural-body/" target="_blank"> here </a>
- Use `pip install neural-body`
- Use the `neural_body` command to run the simulator from terminal.

---
## Usage
This selection includes an overview of all menu buttons and functionality of the simulator.

### Pause / Play
The simulation can be paused at any point.
![Setup Overview GIF](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/play_pause.GIF?raw=true)
### Toggle View
The simulation view can be toggled from overhead to side view.
![Setup Overview GIF](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/toggle_view.GIF?raw=true)
### Adjust Speed
The simulation can be sped up or slowed down.
![Setup Overview GIF](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/adjust_speed.GIF?raw=true)
### New Simulation
The initial state of all bodies in the system are contained in CSV files packaged
with the simulator.  Whichever planet is designated as the "satellite" tells the 
simulator which neural network to use for planetary motion prediction.  This early 
demo can only predict the motion of Mars or Pluto given the positions of the other
planets in the system.  In later releases, the neural network will be updated to 
accommodate predicting the motion of any body.

Current Config File Options:
- mars_sim_config.csv
- pluto_sim_config.csv

![Config File Overview](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/config_overview.png?raw=true)
![Setup Overview GIF](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/new_simulation.GIF?raw=true)
### Is NASA Right?
If you disagree with NASA, you can bring Pluto back as a planet.  
![Setup Overview GIF](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/is_nasa_right.GIF?raw=true)
### Show Planet Key
Hovering over this option displays a color coded key of all planets in the system.
![Setup Overview GIF](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/show_planet_key.GIF?raw=true)
### Travel to a Day
Selecting this option allows the user to rewind or fast forward the simulation by 
entering the day they would like to jump to.  There is a heavy delay for fast forwarding
as the simulator right now must inefficiently calculate every frame between the current
day and the day you entered.  Negative time values will be treated as reverting 
back to 0 day.
![Setup Overview GIF](https://github.com/nedgar76/neural-body/blob/demo-sim/0_demo-sim/readme_resources/travel_to_a_day.GIF?raw=true)

---
## Documentation
To view the documentation online, go to the following URL: \
<a href="https://nedgar76.github.io/neural-body/" target="_blank"> Neural Body Sphinx Documentation </a>

Documentation for the source compiled with Sphinx is included in the `neural-body/0_demo-sim/docs/_build/html/`
folder.  You will need to download and host yourself by running `python3 -m http.server --directory _build/html`
---
# Simulator and Neural Net Improvements Since Alpha

The simulator was originally base on the BenrulesRealTimeSim class, which creates a real time
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
   the next state, we must know the previous state), the team chose to address
   performance with a producer/consumer model that uses multiprocessing to
   continually run a producer in a background process that calculates future
   time steps even while the simulation is paused or when there is available
   CPU time on another processor.  Main simulation calculations are moved to
   an external process that calculates future time steps.  This process
   maintains a pre-queue of about 5000 future time steps.  It feeds a queue
   of 100 time steps between the background process and the main simulator that
   provides positions to the front end.  In order to reliably keep the main
   queue filled, the background process continually checks the main
   queue is filled with values from the pre-queue.  It is able to continually
   perform this task since the calculations filling the pre-queue are launched
   and performed in a separate thread.  This help mitigate the
   degradation of the buffer that the main queue provides.

In addition to the above improvements, benchmarking and queue monitoring are used
to ensure that the system runs at a steady pace that the simulator can keep
up with frame requests.  The user can burst to 2X or 4X, but if the queue begins to degrade,
the simulator will automatically revert back to 1X.  This is mainly for when
the neural network is used.  When not using the neural network, the simulator
will usually set to a max framerate of 50 fps.
---
## Team
The AstroGators formed as a result of the "CIS4930 - Performant Python Programming" 
course at the University of Florida developed by 
<a href="https://www.eng.ufl.edu/eed/faculty-staff/jeremiah-blanchard/" target="_blank"> Jeremiah Blanchard </a>

Team members include:
- Nathaniel Edgar
- Craig Boger
- Gary Jones
- Cory Robertson
- Andrew Sowinski
 
The Github repo for this initial demonstration is located at: 
<a href="https://github.com/nedgar76/neural-body/tree/demo-sim/0_demo-sim" target="_blank"> https://github.com/nedgar76/neural-body/tree/demo-sim/0_demo-sim </a>

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 ©