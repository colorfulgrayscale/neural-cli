#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
from utilities import (DataLoader, DataCleaner, li,
        crossValidateIndices, comparePrediction)
from nn import MLP
import sys
import pandas as pd
import numpy as np

#sys.tracebacklimit = 0

# TODO get the argparser stuff to work
parser = argparse.ArgumentParser(description='nueral-cli')


def crossValidate(data, meta, folds=5):
    "k-fold cross validation"
    for counter, (training, validation) in enumerate(crossValidateIndices(items=range(data.shape[0]), k=folds, randomize=True)):
        # setup training and validation matricies
        train               = data.ix[training].reset_index(drop=True)
        validate            = data.ix[validation].reset_index(drop=True)
        trainingFeatures    = train.drop(meta.categoricalLabelColumns, axis=1)  # remove output columns
        validationFeatures  = validate.drop(meta.categoricalLabelColumns, axis=1)  # remove output columns
        trainingLabels      = train[meta.categoricalLabelColumns]  # use only output columns
        validationLabels    = validate[meta.categoricalLabelColumns]  # use only output columns

        #setup MLP and start training
        mlp = MLP()
        li("Fold {2}/{3} - Training with {1}/{4} rows ({0} iterations)".format(mlp.trainingIterations, trainingFeatures.shape[0], counter + 1, folds, data.shape[0]))
        mlp.train(trainingFeatures, trainingLabels, topology="5")

        #make predictions
        li("Fold {0}/{1} - Testing with {2}/{3} rows".format(counter + 1, folds, validationFeatures.shape[0], data.shape[0]))
        predictedOutput = list()
        for loopCounter, (rowIndex, row) in enumerate(validationFeatures.iterrows()):
            predictedOutput.append([item for item in mlp.predict(row)])
        predictedOutput = pd.DataFrame(np.array(predictedOutput), columns=[meta.categoricalLabelColumns])

        predictedMatrix = DataCleaner.deNormalize(DataCleaner.categoricalToNominal(validationFeatures.join(predictedOutput), meta), meta)
        expectedMatrix = DataCleaner.deNormalize(DataCleaner.categoricalToNominal(validationFeatures.join(validationLabels), meta), meta)
        comparePrediction(predictedMatrix, expectedMatrix, meta)


if __name__ == '__main__':

    loader = DataLoader(sys.argv[1])
    li("Loading Dataset")
    data, meta = loader.load()
    li(meta)
    li("Dataset has Rows: {0}, Columns: {1}".format(len(data), len(meta.names())))

    labelID = raw_input('Which columns to use as labels? [{0}-{1}]: '.format(1, len(meta.names())))
    labelID = labelID.split(",")
    meta.labelColumns = []
    meta.categoricalLabelColumns = []
    for index in labelID:
        try:
            colName = data.icol(int(index) - 1).name
        except:
            raise Exception("Invalid column index entered")
        else:
            meta.labelColumns.append(colName)
            meta.categoricalLabelColumns.append(colName)
    if len(meta.labelColumns) < 1:
        raise Exception("Specify atleast 1 label column")

    li("Cleaning Data")
    data = DataCleaner.impute(data, meta)
    data = DataCleaner.normalize(data, meta)
    data = DataCleaner.nominalToCategorical(data, meta)

    li("Starting fold cross validation")
    crossValidate(data, meta)
