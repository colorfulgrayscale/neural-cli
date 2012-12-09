#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
from utilities import (DataLoader, DataCleaner, li,
        crossValidateIndices, setLabels)
from nn import MLP

#sys.tracebacklimit = 0

parser = argparse.ArgumentParser(description='Commandline neural network utility')
parser.add_argument('-d', '--data', help='Dataset to use')
parser.add_argument('-l', '--labels', help='Which columns to use as labels')

subparsers = parser.add_subparsers(dest='command', help="Specify one of these actions")
parser_cv = subparsers.add_parser('cross_validate')
parser_cv.add_argument('-k', '--folds', help='How many times to do cross validation ', default=3, type=int)
parser_cv.add_argument('-g', '--graph', help='Graph the learning Process and save it to specified file name')
parser_cv.add_argument('-t', '--topo', help='Neural Network Topology', default="5")
parser_cv.add_argument('-e', '--epochs', help='# of training iterations over entire dataset', default=400, type=int)
parser_cv.add_argument('-w', '--weights', help='Initial Weights Multiplier', default=1.0, type=float)

parser_train = subparsers.add_parser('train')
parser_train.add_argument('-o', '--output', help='Output trained model to a JSON')
parser_train.add_argument('-g', '--graph', help='Graph the learning Process and save it to specified file name')
parser_train.add_argument('-t', '--topo', help='Neural Network Topology', default="5")
parser_train.add_argument('-e', '--epochs', help='# of training iterations over entire dataset', default=400, type=int)
parser_train.add_argument('-w', '--weights', help='Initial Weights Multiplier', default=1.0, type=float)


parser_predict = subparsers.add_parser('predict')
parser_predict.add_argument('-i', '--input', help='Trained model, JSON')

args = vars(parser.parse_args())
print args


def crossValidate(data, meta, folds, topology, iterations, weights, graph=None):
    "k-fold cross validation"
    if folds <= 1:
        raise Exception("Cross validation folds must be > 1")
    averageError = 0.0
    for counter, (training, validation) in enumerate(crossValidateIndices(items=range(data.shape[0]), k=folds, randomize=True)):
        # setup training and validation matricies
        train               = data.ix[training].reset_index(drop=True)
        validate            = data.ix[validation].reset_index(drop=True)
        trainingFeatures    = train.drop(meta.categoricalLabelColumns, axis=1)  # remove output columns
        validationFeatures  = validate.drop(meta.categoricalLabelColumns, axis=1)  # remove output columns
        trainingLabels      = train[meta.categoricalLabelColumns]  # use only output columns
        validationLabels    = validate[meta.categoricalLabelColumns]  # use only output columns

        #setup MLP and start training
        li("Fold {2}/{3} - Training with {1}/{4} rows ({0} epochs)".format(iterations, trainingFeatures.shape[0], counter + 1, folds, data.shape[0]))
        mlp                          = MLP()
        mlp.trainingIterations       = iterations
        mlp.initalWeightsMultiplier  = weights
        mlp.features                 = trainingFeatures.values  # convert from pandas dataframe to numpy arrays. they are faster for the computationally intensive training phase.
        mlp.labels                   = trainingLabels.values
        mlp.validationFeatures       = validationFeatures  # for validation, send in pandas dataframes.
        mlp.validationLabels         = validationLabels
        mlp.meta                     = meta
        mlp.topology                 = topology
        if graph:
            mlp.trackLearning            = True
        mlp.setupHiddenLayers()
        mlp.train()

        if graph:
            li("Plotting Learning to file '{0}'".format(graph))
            mlp.plotLearning(graph)

        #validate model
        li("Fold {0}/{1} - Testing with {2}/{3} rows".format(counter + 1, folds, validationFeatures.shape[0], data.shape[0]))
        error = mlp.validateModel(printToScreen=True)
        averageError += error

    averageError = averageError / folds
    li("Average error across all folds: {0}".format(averageError))
    return averageError


if __name__ == '__main__':

    if not args['data']:
        data = raw_input('Path to dataset: ')
    else:
        data = args['data']

    loader = DataLoader(data)
    li("Loading Dataset")
    data, meta = loader.load()
    li(meta)
    li("Dataset has Rows: {0}, Columns: {1}".format(len(data), len(meta.names())))

    if not args['labels']:
        labels = raw_input('Which columns to use as labels? [{0}-{1}]: '.format(1, len(meta.names())))
    else:
        labels = args['labels']
    setLabels(data, meta, labels)
    if len(meta.labelColumns) < 1:
        raise Exception("Specify atleast 1 label column")
    li("Label Columns: {0}".format(meta.labelColumns))

    li("Cleaning Data")
    data = DataCleaner.impute(data, meta)
    data = DataCleaner.normalize(data, meta)
    data = DataCleaner.nominalToCategorical(data, meta)

    if args['command'] in ("cross_validate", "train"):
        if args['command'] == "cross_validate":
            li("Starting {0}-fold cross validation".format(args['folds']))
            error = crossValidate(data, meta, folds=args['folds'], topology=args['topo'], iterations=args['epochs'], weights=args['weights'], graph=args['graph'])

    #with open("~/nncli-history.txt", "a") as myfile:
        #myfile.write("\n{0},{1},{2},{3}".format(args['data'], args['command'], args['topo'], error))
