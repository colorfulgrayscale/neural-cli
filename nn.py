#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math
import pandas as pd
from collections import OrderedDict
from progressbar import ETA, ProgressBar, Bar, Percentage
import matplotlib.pyplot as plt
import matplotlib
from types import NoneType
from utilities import DataCleaner, comparePrediction

#progress bar formatting
widgets = [Percentage(), " ", Bar('|'), ' ', ETA()]

# TODO softmax and possibly linear
activationFunctions = {
        'logistic': lambda x: 1 / (1 + math.exp(-x)),
        'tanh': lambda x: math.tanh(x),
        }
activationFunctionsD = {
        'logistic': lambda x: math.exp(x) / ((math.exp(x) + 1) ** 2),
        'tanh': lambda x: (math.sech(x)) ** 2,
        }


class Layer():
    """
    Represents each layer in a multilayer preceptron.
    Contains weights, inputs, outputs for each Layer
    """

    def __init__(self, rows, columns, name="Hidden Layer", activation="logistic", weightsMultiplier=1):
        """ columns - neurons in current layer,  rows - size of the previous layer """
        self.weights = np.random.standard_normal(size=(rows, columns)) * weightsMultiplier  # initialize weights matrix. size of weights manipulated with the multiplier.
        self.bias = (np.zeros(shape=(rows)))  # bias vector initialized with zeros
        self.inputs = (np.zeros(shape=(rows)))  # input vector
        self.outputs = (np.zeros(shape=(rows)))  # output vector
        self.error = (np.zeros(shape=(rows)))  # error vector
        self.name = name  # name of the layer, for indexing
        self.neuronCount = rows
        self.weightsCount = columns
        self.setActivation(activation)  # set activation function for neuron

        # variables to store snapshots of weights over time
        # allows to vizualize how the weights as the network trains
        self.weightsHistory = []

    def setActivation(self, f):
        """ Set activation function for layer. 'f' Must be sigmoid """
        self.activation = activationFunctions[f]  # lambda of activation function
        self.activationD = activationFunctionsD[f]  # lambda of derivative of the activation function

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
        self.learningRate = 0.1
        self.trainingIterations = 400  # TODO autodetect convergence
        self.initalWeightsMultiplier = 1  # A multiplier to experiment with different intial weight configurations
        self.features = None
        self.labels = None
        self.validationFeatures = None
        self.validationLabels = None
        self.preceptronLayers = None
        self.meta = None  # data about dataset
        self.topology = "?"  # default topology, no hidden layers

        # track accuracy and error over the process of training
        self.trackLearning = False
        self.errorHistoryX = list()
        self.errorHistoryY = list()

    def setupHiddenLayers(self):
        """ Allocate Hidden Layers based on Specified Topology. Auto detects input and output layers """
        if NoneType in (type(self.features), type(self.labels)):
            raise Exception("Specify features, labels before allocating toplogy")
        #parse topology
        if '?' in self.topology:
            layerUnits = list()  # no topo specified
        else:
            layerUnits = self.topology.split("-")  # parse given topo
        layerUnits.insert(0, self.features.shape[1])  # how many feature attributes
        try:
            layerUnits.append(self.labels.shape[1])  # how many label attributes
        except IndexError:
            layerUnits.append(self.labels.shape[0])  # if its only 1d, switch to 0
        layerUnits = [int(x) for x in layerUnits]
        self.preceptronLayers = OrderedDict()  # ordered dictionary to keep track of precetrons and for easy lookup by name

        #alocate toplogy
        for value in xrange(1, len(layerUnits)):
            layer = Layer(layerUnits[value], layerUnits[value - 1], weightsMultiplier=self.initalWeightsMultiplier)
            if value == len(layerUnits) - 1:
                layer.name = "Output Layer"
            else:
                layer.name = "Hidden Layer: {0}".format(value)
            self.preceptronLayers[layer.name] = layer  # add it to a local datastucture to keep track of all layers

    def printPreceptrons(self):
        """ Print all layers. For Debugging """
        for j in xrange(0, len(self.preceptronLayers)):
            print self.preceptronLayers.values()[j]
        print "\n"

    def validateModel(self, printToScreen=False):
        if NoneType in (type(self.validationFeatures), type(self.validationLabels), type(self.meta)):
            raise Exception("Specify validation features and labels")

        predictedOutput = list()
        for loopCounter, (rowIndex, row) in enumerate(self.validationFeatures.iterrows()):
            predictedOutput.append([item for item in self.predict(row)])
        predictedOutput = pd.DataFrame(np.array(predictedOutput), columns=[self.meta.categoricalLabelColumns])

        predictedMatrix = DataCleaner.deNormalize(DataCleaner.categoricalToNominal(self.validationFeatures.join(predictedOutput), self.meta), self.meta)
        expectedMatrix = DataCleaner.deNormalize(DataCleaner.categoricalToNominal(self.validationFeatures.join(self.validationLabels), self.meta), self.meta)
        error = comparePrediction(predictedMatrix, expectedMatrix, self.meta, printToScreen)
        return error

    def train(self, topology="?-?"):
        if NoneType in (type(self.features), type(self.labels), type(self.preceptronLayers)):
            raise Exception("Specify features, labels and allocate hidden layers before training")

        pbar = ProgressBar(widgets=widgets)
        for i in pbar(xrange(self.trainingIterations)):
            if self.trackLearning:
                if i % 10 == 0:
                    for layerName in self.preceptronLayers.keys():
                        layer = self.preceptronLayers[layerName]
                        layer.weightsHistory.append(np.sqrt((layer.weights ** 2).sum(axis=0).sum()))
                    self.errorHistoryY.append(self.validateModel())
                    self.errorHistoryX.append(i)
            for rowIndex in xrange(self.features.shape[0]):
                feature = self.features[rowIndex]
                label = self.labels[rowIndex]
                self.predict(feature)  # make prediction
                self.computeErrors(label)  # compute error and back propogate to all layers
                self.recalculateWeights(feature)  # recompute weights based on errors

            self.learningRate *= 0.997  # attenuate learning rate

    def plotLearning(self, filename):
        """ Plot evolution of weights through the training """
        if not self.trackLearning:
            raise Exception("cannot plot, no tracking data")
        font = {'size' : 10}
        matplotlib.rc('font', **font)
        plt.figure()
        plt.subplot(211)
        plt.title("Evolution of Error and Weights")
        plt.plot(self.errorHistoryX, self.errorHistoryY, label='Error')
        plt.ylabel('prediction error')
        plt.grid(True)
        plt.ylim(bottom=0.0, top=1.0)
        plt.subplot(212)
        for layerName in self.preceptronLayers.keys():
            layer = self.preceptronLayers[layerName]
            plt.plot(self.errorHistoryX, layer.weightsHistory, label='{0}'.format(layerName))
        plt.ylabel('root sum square of weights')
        plt.grid(True)
        plt.legend(loc="best", prop={'size': 6})
        plt.xlabel('epochs')
        plt.savefig(filename)

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
