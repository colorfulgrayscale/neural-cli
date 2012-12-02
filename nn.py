#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Multi Layer Preceptron - Neural Network
"""
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from collections import OrderedDict
from progressbar import ETA, ProgressBar, Bar, Percentage


class Layer():
    """
    Represents each layer in a multilayer preceptron.
    Contains weights, inputs, outputs for each Layer
    """

    def __init__(self, rows, columns, name="Hidden Layer"):
        """ columns - neurons in current layer,  rows - size of the previous layer """
        self.weights = (np.random.standard_normal(size=(rows, columns)) / 10)  # initialize weights matrix to small random values
        self.bias = (np.zeros(shape=(rows)))  # bias vector initialized with zeros
        self.inputs = (np.zeros(shape=(rows)))  # input vector
        self.outputs = (np.zeros(shape=(rows)))  # output vector
        self.error = (np.zeros(shape=(rows)))  # error vector
        self.name = name  # name of the layer, for indexing
        self.activation = None  # lambda of the activation function
        self.neuronCount = rows
        self.weightsCount = columns

    def setActivation(self, x, f):
        """ Set activation function for layer. 'f' Must be sigmoid """
        self.activation = lambdify(x, f)  # lambda of activation function
        self.activationD = lambdify(x, sp.diff(f, x))  # lambda of derivative of the activation function

    def pickle(self):
        """ Serialize Layer """
        return (self.weights, self.bias, self.inputs,
                self.outputs, self.error, self.name,
                self.neuronCount, self.weightsCount)

    def unPickle(self, attribs):
        """ Deserialize Layer """
        self.weights = attribs[0]
        self.bias = attribs[1]
        self.inputs = attribs[2]
        self.outputs = attribs[3]
        self.error = attribs[4]
        self.name = attribs[5]
        self.neuronCount = attribs[6]
        self.weightsCount = attribs[7],

    def __repr__(self):
        """ To string method. For debugging. """
        o = str(self.name) + "\n"
        o += "-" * 15  + "\n"
        o += "Weights: " + str(self.weights.shape) + "\n" + str(self.weights) + "\n\n"
        o += "Bias: " + str(self.bias.shape) + "\n" + str(self.bias) + "\n\n"
        o += "Inputs: " + str(self.inputs.shape) + "\n" + str(self.inputs) + "\n\n"
        o += "Outputs: " + str(self.outputs.shape) + "\n" + str(self.outputs) + "\n\n"
        o += "Error: " + str(self.error.shape) + "\n" + str(self.error) + "\n\n"
        return o


class MLP():
    """ Multi Layer Preceptron """

    def __init__(self):
        self.features = None
        self.labels = None
        self.preceptronLayers = OrderedDict()  # ordered dictionary to keep track of precetrons and for easy lookup by name
        self.learningRate = 0.1
        self.trainingIterations = 500  # TODO autodetect convergence
        self.trained = False

    def train(self, features, labels, topology="?-?"):
        self.features = features.values  # convert pandas dataframe to numpy array for speed
        self.labels = labels.values

        if '?' in topology:
            layerUnits = list()  # no topo specified
        else:
            layerUnits = topology.split("-")  # parse given topo
        layerUnits.insert(0, features.shape[1])  # how many feature attributes
        try:
            layerUnits.append(labels.shape[1])  # how many label attributes
        except IndexError:
            layerUnits.append(labels.shape[0])  # if its only 1d, switch to 0

        self.allocateTopology([int(x) for x in layerUnits])  # pass a list of integers

        widgets = [Percentage(), " ", Bar('|'), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets)  # Progressbar can guess maxval automatically.
        for i in pbar(xrange(self.trainingIterations)):
            for rowIndex in xrange(self.features.shape[0]):
                feature = self.features[rowIndex]
                label = self.labels[rowIndex]
                self.predict(feature)  # make prediction
                self.computeErrors(label)  # compute error and back propogate to all layers
                self.recalculateWeights(feature)  # recompute weights based on errors
            self.learningRate *= 0.997  # attenuate learning rate

        self.trained = True  # a bit to tell if the model is trained or not

    def allocateTopology(self, layerUnits):
        """ Allocate Layers based on Specified Topology """
        for value in xrange(1, len(layerUnits)):
            layer = Layer(layerUnits[value], layerUnits[value - 1])
            if value == len(layerUnits) - 1:
                layer.name = "Output Layer"
            else:
                layer.name = "Hidden Layer: {0}".format(value)
            x = sp.symbols('x')
            logistic = 1 / (1 + (sp.E ** -x))  # logistic function
            layer.setActivation(x, logistic)  # set activation function for neuron
            self.preceptronLayers[layer.name] = layer  # add it to a local datastucture to keep track of all layers

    def printPreceptrons(self):
        """ Print all layers. For Debugging """
        for j in xrange(0, len(self.preceptronLayers)):
            print self.preceptronLayers.values()[j]
        print "\n"

    def predict(self, feature):
        """ Predict, given feature vector"""
        # loop through the layers and compute input and output for each neuron
        for i in xrange(-1, len(self.preceptronLayers) - 1):
            currentLayer = self.preceptronLayers.values()[i + 1]
            if i == -1:  # if this is the first layer, the the previous layer is the input
                previousLayerOutputs = feature
            else:  # else, get previous outputs from the previous layer
                previousLayerOutputs = self.preceptronLayers.values()[i].outputs
            for k in xrange(currentLayer.neuronCount):
                currentLayer.inputs[k] = np.dot(previousLayerOutputs, currentLayer.weights[k]) + currentLayer.bias[k]
                currentLayer.outputs[k] = currentLayer.activation(currentLayer.inputs[k])

        return self.preceptronLayers['Output Layer'].outputs

    def computeErrors(self, label):
        """ Compute errors for each layer based on expected values"""
        #compute error for output layer
        outputLayer = self.preceptronLayers['Output Layer']
        for k in xrange(outputLayer.neuronCount):
            outputLayer.error[k] = (label[k] - outputLayer.outputs[k]) * outputLayer.activationD(outputLayer.inputs[k])

        #Backpropogate and calculate error for each hidden layer based on error in output layer
        for i in reversed(xrange(0, len(self.preceptronLayers) - 1)):
            currentLayer = self.preceptronLayers.values()[i]
            nextLayer = self.preceptronLayers.values()[i + 1]
            for k in xrange(currentLayer.neuronCount):  # for each neuron in layer, compute error
                currentLayer.error[k] = np.dot(nextLayer.error, nextLayer.weights.T[k]) * currentLayer.activationD(currentLayer.inputs[k])

        return outputLayer.error

    def recalculateWeights(self, feature):
        """ Recalculate Weights for each layer of the preceptron based on error """
        for i in reversed(xrange(0, len(self.preceptronLayers))):
            currentLayer = self.preceptronLayers.values()[i]
            if i == 0:  # if this is the first layer, set previous layer as the input features
                previousLayerOutputs = feature
            else:  # if not, just get outputs from previous preceptron layer
                previousLayerOutputs = self.preceptronLayers.values()[i - 1].outputs
            gradient = np.array([(error * previousLayerOutputs) for error in currentLayer.error])
            currentLayer.weights += gradient * (self.learningRate)  # weights gradient
            currentLayer.bias += currentLayer.error * (self.learningRate)  # bias gradient

    def updateInputs(self, feature):
        """ Update Features based on error and weights """
        currentLayer = feature
        nextLayer = self.preceptronLayers.values()[0]
        for k in xrange(len(currentLayer)):  # for each neuron in layer, subtract gradient
            gradient = np.dot(nextLayer.error, nextLayer.weights.T[k])
            currentLayer[k] += gradient * (self.learningRate)
        return currentLayer
