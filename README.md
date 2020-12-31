# MNIST Neural Network
---
A neural network is a system of interconnected nodes, or artificial neurons that perform some task by learning from a dataset and incrementally improving its own performance. These artificial neurons are organised into multiple layers including an input layer, where data is fed forward through the network's successive layers, until it produces some output in the final layer.

Networks "learn" by analyzing a dataset of training inputs, where each training example is classified by a label. Through a process called backpropagation, the network adjusts the "weights" connecting each neuron (which can be thought of as the synapses connecting neurons in a human brain) based on how close the output produced from traning examples, which classifies each training example, is to the actual classification of those examples. Biases for each neuron are also updated accordingly.

### The MNIST Dataset
This project produces a neural network that classifies images of handwritten digits ranged from 0-9. These images are gathered from the MNIST database - a large set of images of handwritten digits commonly used for training neural networks like this one. This is my first attempt at building a neural network from scratch and I plan to continually update this project as I improve my code.

### Installation
To install this project, clone the repository onto your computing using git bash and the command:
> git clone https://github.com/DavidOWade/MNIST-Neural-Network.git

Using anaconda powershell, or any utility that can run Jupyter Notebook, navigate to the folder holding the cloned repository, and run the command: 
> jupyter lab
