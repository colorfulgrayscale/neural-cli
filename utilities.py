#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.io.arff.arffread import MetaData
from collections import defaultdict
from matplotlib.mlab import csv2rec
from random import shuffle

#setup logging
import logging
log = logging.getLogger('pynu')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
ch.setFormatter(formatter)
log.addHandler(ch)
li = log.info
ld = log.debug


class DataLoader:
    """Loads data from various file formats into a Pandas Dataframe"""

    def __init__(self, fileName):
        if not os.path.isfile(fileName):
            raise IOError("'{0}' not found.".format(fileName))
        self.fileExtension = os.path.splitext(fileName)[1]
        self.fileName = fileName
        self.baseName = os.path.basename(fileName)

    def load(self, header=False):
        """ Load file based on extension """
        if self.fileExtension in (".csv", ".data", ".txt", ".gz"):
            data, meta = self._loadCSV(header)
        elif self.fileExtension in (".arff", ".arf"):
            data, meta = self._loadARFF()
        else:
            raise Exception("Unknown file format. Currently supports csv, txt, gz and arff")
        # set some labels meta
        meta.labelColumns = []  # store the label fields
        meta.categoricalLabelColumns = []
        return data, meta

    def extractMeta(self, data):
        """ Attempts to reverse engineer meta data from pandas dataframe"""
        meta = []
        for counter, column in enumerate(data.columns):
            series = data[column]
            if series.dtype == np.dtype('object'):
                enums = series.unique().tolist()
                if "" in enums:
                    enums.remove("")  # remove blanks in the enums if they exist
                if len(enums) <= 1:
                    raise Exception("Not enough info in the data set to automatically extract meta data. Failed at column {0}".format(counter + 1))
                meta.append((column, "{{{0}}}".format(",".join(enums))))
            else:
                meta.append((column, 'REAL'))
        return MetaData(self.baseName, meta)

    def _loadCSV(self, header=False):
        "Plain text files"
        if header:
            header = None
        else:
            header = ",".join("C{0}".format(i) for i in range(1, 90))
        data = pd.DataFrame(csv2rec(self.fileName, names=header, missing="?"))
        meta = self.extractMeta(data)
        return data, meta

    def _loadARFF(self):
        "*.ARFF, WEKA files"
        data, meta = arff.loadarff(self.fileName)
        return pd.DataFrame(data), meta

    # TODO add loaders for HDF5, Excel and Matlab '.mat' formats


class DataCleaner:
    """Utilities to clean data in a dataframe"""

    @staticmethod
    def impute(inputDataFrame, meta):
        """
        Fill in missing values in data.
        For continuous attributes, replace missing values with the column mean.
        For categorical attributes, replace missing values with the mode of the categorical distribution (the most common value).
        """
        dataFrame = inputDataFrame.copy()  # make a copy, dont screw with the original
        for columnName in meta.names():
            columnType, columnCategories = meta[columnName]
            column = dataFrame[columnName]
            if columnType == "nominal":
                # if its a nominal column, replace unknowns with most occuring element
                valuesCount = column.value_counts()  # a sorted list containing the frequency of occurances of all values
                if '' in valuesCount.keys() or '?' in valuesCount.keys():  # check if there are any unknown values in column
                    mode = valuesCount.index[0]  # pick the top most element from value count
                    if mode in ("", "?"):
                        mode = valuesCount.index[1]  # if the most occuring element happens to be an unknown, just pick the next
                    column[column == ""] = mode
                    column[column == "?"] = mode
                    dataFrame[columnName] = column  # update data frame with new column
            elif columnType == "numeric":
                # if its a numeric column, replace '?' with mean of column
                if column.dtype == np.dtype('float64'):
                    column[column.isnull()] = column.mean()
                elif column.dtype == np.dtype('int64'):
                    column[column < 0] = column.mean()
                else:
                    raise Exception("{0} Column has an unknown numeric datatype: {0}".format(columnName, column.dtype))
                dataFrame[columnName] = column  # update frame with column
            else:
                raise Exception("Unknown Column Type Detected: ", columnType)
        return dataFrame

    @staticmethod
    def normalize(inputDataFrame, meta):
        """
        This normalizes a matrix so that all continuous values fall from 0 to 1 so that the model will converge faster.
        It leaves nominal values unchanged.
        Computes the column min and the column max of the training data.
        Subtract the min from each value. Then, divide each value by (max - min).
        """
        dataFrame = inputDataFrame.copy()  # make a copy, dont screw with the original
        for columnName in meta.names():
            columnType, columnCategories = meta[columnName]
            column = dataFrame[columnName]

            # create dictionaries to store column min/max
            # these will be later used to denormalize
            if not hasattr(meta, "columnMin"):
                meta.columnMin = {}
            if not hasattr(meta, "columnMax"):
                meta.columnMax = {}

            if columnType == "numeric":  # only interested in numerical columns
                if column.dtype != np.dtype('float64'):
                    column = column.astype(np.dtype('float64'))  # convert it to float before any processing
                if type(inputDataFrame) == pd.DataFrame:  # normalize dataframe
                    meta.columnMin[columnName], meta.columnMax[columnName] = column.min(), column.max()  # store column mean data in meta. We'll need this when we denormalize
                # else if it was a pandas series, it'll use the mean values specified in the meta
                difference = 1.0 * (meta.columnMax[columnName] - meta.columnMin[columnName])
                dataFrame[columnName] = (column - meta.columnMin[columnName]) / difference  # update with normalized value
        return dataFrame

    @staticmethod
    def deNormalize(inputDataFrame, meta):
        """
        This denormalizes a matrix.
        multiply by the range and then add the minimum.
        """
        dataFrame = inputDataFrame.copy()  # make a copy, dont screw with the original
        for columnName in meta.names():
            columnType, columnCategories = meta[columnName]
            try:
                column = dataFrame[columnName]
            except KeyError:
                continue  # if the column dosent exist, just skip it
            if not hasattr(meta, "columnMin") or not hasattr(meta, "columnMin"):
                raise Exception("Column means info not found in meta data")
            if columnType == "numeric":  # only interested in numerical columns
                columnMin, columnMax = meta.columnMin[columnName], meta.columnMax[columnName]
                difference = columnMax - columnMin
                dataFrame[columnName] = (column * difference) + columnMin  # update with denormalized value
        return dataFrame

    @staticmethod
    def nominalToCategorical(inputDataFrame, meta):
        """
        This converts nominal attributes to a categorical distribution for each column
        y in {"u", "y", "l", "t"} -> <0,1,0,0>
        We do this because a NN cannot understand categorical values
        """
        dataFrame = inputDataFrame.copy()  # make a copy, dont screw with the original
        # if its a pandas series,
        if type(inputDataFrame) == pd.Series:
            for columnName in meta.names():
                column = dataFrame[columnName]
                columnType, columnCategories = meta[columnName]
                if columnType == "nominal":  # only interested in numerical columns
                    try:
                        categoryIndex = columnCategories.index(column)
                    except ValueError:
                        raise Exception("Unknown Category Value Detected: ", column)
                    binaryVector = np.zeros(len(columnCategories))
                    binaryVector[categoryIndex] = 1
                    # do nomcat and create new columns
                    for counter, value in enumerate(binaryVector):
                        key = '{0}-{1}'.format(columnName, counter)
                        dataFrame = dataFrame.append(pd.Series({key: value}))
                    # remove original column
                    dataFrame = dataFrame.drop(columnName)
            return dataFrame

        # if its a pandas dataframe

        if "nominal" not in meta.types():
            return inputDataFrame  # if there are no nomial values, return

        newDataFrame = pd.DataFrame()
        meta.nomCatMappings = dict()  # for storing nominalToCategorical column mappings
        for columnName in meta.names():
            columnType, columnCategories = meta[columnName]
            column = dataFrame[columnName]
            if columnType == "nominal":  # only interested in numerical columns
                #each nominal column is broken down into subcolumns of len(all possible values)
                #store column disassociations in metadata. This data will be used to convert back to nominal values
                subColumns = ['{0}-{1}'.format(columnName, x) for x in range(len(columnCategories))]
                meta.nomCatMappings[columnName] = subColumns

                if hasattr(meta, "categoricalLabelColumns"):
                    # update output column name
                    labelColumns = getattr(meta, "categoricalLabelColumns")
                    try:
                        columnID = [i for i, x in enumerate(labelColumns) if x == columnName][0]
                    except:
                        pass
                    else:
                        labelColumns.pop(columnID)
                        for item in reversed(subColumns):
                            labelColumns.insert(columnID, item)

                columnBinaryList = list()  # stores a list of binary value corresponding to an enum for each column
                for counter, value in enumerate(column):
                    try:
                        categoryIndex = columnCategories.index(value)  # get index of value in enum list
                    except ValueError:
                        raise Exception("Unknown Category Value Detected: '{0}' in Column '{2}'. Was expecting [{1}]".format(value, ",".join(i for i in columnCategories), columnName))
                    binaryVector = np.zeros(len(columnCategories))
                    binaryVector[categoryIndex] = 1
                    columnBinaryList.append([item for item in binaryVector])  # expand column to span several subcolumns
                #after every row in the column has been converted into a binary vector, push it into a dataframe
                columnBinaryList = pd.DataFrame(np.array(columnBinaryList), columns=subColumns)
            elif columnType == "numeric":
                #if numeric, then simply push it into a new dataframe, no processing required
                columnBinaryList = pd.DataFrame(column)
            # if dataframe is empty, set our new dataframe to it. If not, append to it
            if newDataFrame.empty:
                newDataFrame = columnBinaryList.copy()
            else:
                newDataFrame = newDataFrame.join(columnBinaryList)
        return newDataFrame

    @staticmethod
    def categoricalToNominal(inputDataFrame, meta):
        """
        This converts from categorical distribution back into its original Nominal form
        <0,1,0,0> -> y in {"u", "y", "l", "t"}
        """
        dataFrame = inputDataFrame.copy()  # make a copy, dont screw with the original
        newDataFrame = pd.DataFrame()

        if "nominal" not in meta.types():
            return inputDataFrame  # if there are no nomial values, return

        if not hasattr(meta, "nomCatMappings"):
            raise Exception("Mappings info not found in metadata")

        for subColumnName in dataFrame:
            try:  # if this subcolumn has a parent column, find it
                columnName = (key for key, value in meta.nomCatMappings.items() if subColumnName in value).next()  # reverse dictionary lookup
            except StopIteration:  # if not, then column name is same as name of subcolumn
                columnName = subColumnName
            if columnName in list(newDataFrame.keys()):
                continue  # if we've already processed this column's parent, skip iteration
            columnType, columnCategories = meta[columnName]
            column = dataFrame[subColumnName]
            if columnType == "nominal":  # only interested in nominal columns
                subColumns = meta.nomCatMappings[columnName]
                columnNominalValues = list()
                for loopCounter, (rowIndex, row) in enumerate(dataFrame[subColumns].iterrows()):  # iterate only over the subcolumns and try to combine them
                    index = row.argmax()  # index of the maximum value in row
                    columnNominalValues.append(columnCategories[index])  # add this to a new column
                columnNominal = pd.DataFrame(np.array(columnNominalValues), columns=[columnName])  # create a new data frame from the created columns
            elif columnType == "numeric":
                columnNominal = pd.DataFrame(column)  # numerical values are passed on untouched
            if newDataFrame.empty:  # if dataframe is empty, set our new dataframe to it. If not, append to it
                newDataFrame = columnNominal.copy()
            else:
                newDataFrame = newDataFrame.join(columnNominal)
        return newDataFrame


def setLabels(data, meta, labels):
    """ Set the label columns in a dataset"""
    labelID = labels.split(",")
    for index in labelID:
        index = index.strip()
        if index.isdigit():
            try:
                colName = data.icol(int(index) - 1).name
            except:
                raise Exception("Invalid column index entered")
            else:
                meta.labelColumns.append(colName)
                meta.categoricalLabelColumns.append(colName)
        else:
            try:
                colName = data[index]
            except:
                raise Exception("Invalid column name entered")
            else:
                meta.labelColumns.append(index)
                meta.categoricalLabelColumns.append(index)


def crossValidateIndices(items, k, randomize=False):
    """adapted from http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/#c1"""
    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in xrange(k)]

    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation


def comparePrediction(predictedMatrix, expectedMatrix, meta, printToScreen=False):
    """ Annotate predictions by comparing it to expected results"""
    annotatedMatrix = predictedMatrix.copy()
    labelCount = len(meta.labelColumns)
    errorDisplay = defaultdict(list)
    columnErrors = defaultdict(float)
    rowErrors = []
    for loopCounter, (rowIndex, row) in enumerate(predictedMatrix.iterrows()):
        rowError = 0.0
        for labelColumn in meta.labelColumns:
            prediction = row[labelColumn]
            expected = expectedMatrix.ix[rowIndex][labelColumn]
            columnType, columnCategories = meta[labelColumn]
            if columnType == "nominal":  # calculate hamming distance
                if prediction == expected:
                    errorDisplay[labelColumn].append("✓")
                    error = 0.0
                else:
                    errorDisplay[labelColumn].append("✖ ({0})".format(expected))
                    error = 1.0
            elif columnType == "numeric":
                error = expected - prediction
                errorDisplay[labelColumn].append(error)
            else:
                raise Exception("Unknown Column Type: {0}".format(columnType))
            squareError = error * error
            columnErrors[labelColumn] += squareError  # column error = sum of squared errors
            if labelCount > 1:
                rowError += squareError

        if labelCount > 1:  # calculate total error only if there are more than 2 columns
            rowErrors.append(math.sqrt(rowError))  # total error = √(column1Error² + column2Error² + columnNError²)
            rowError = 0.0

    errorStats = [""]
    totalError = 0.0
    for column in errorDisplay.keys():
        annotatedMatrix = annotatedMatrix.join(pd.DataFrame(errorDisplay[column], columns=['{0}Error'.format(column), ]))
        columnType, columnCategories = meta[column]
        if columnType == "nominal":
            correct = round(((predictedMatrix.shape[0] - columnErrors[column]) / float(predictedMatrix.shape[0])) * 100, 2)
            # root mean square error
            e = math.sqrt(columnErrors[column] / float(predictedMatrix.shape[0]))
            errorStats.append("'{0}' Root Mean Square Error: {1} ({2}% accurate)".format(column, e, correct))
        elif columnType == "numeric":
            # root sum squared error
            e = math.sqrt(columnErrors[column])
            errorStats.append("'{0}' Root Sum Square Error: {1}".format(column, e))
        totalError += e

    if labelCount > 1:
        annotatedMatrix = annotatedMatrix.join(pd.DataFrame(rowErrors, columns=['totalError', ]))
        totalError = sum(rowErrors)
        errorStats.append("Total Error: {0}".format(totalError))

    if printToScreen:
        print annotatedMatrix.to_string()
        print "\n".join(errorStats)

    return totalError
