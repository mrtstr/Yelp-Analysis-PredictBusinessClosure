"""

This script reads trains and evaluates a logistic regression model prediction the probability for business closure.
the the prediction is made based of all attributes and on subsets of attributes containing only most important attributes.
plot the ROC curve, feature importance and a report file containing some performance metrics for all of the models

"""

import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.inspection import permutation_importance
import argparse
import json
from pathlib import Path
from matplotlib import rcParams
from tabulate import tabulate
rcParams.update({'figure.autolayout': True})

sns.set_context("talk", font_scale=0.9)

class Model:
    """
    A class containing the model and data sets


    Attributes
    ----------
    x : dataframe
        a dataframe contraining all attributes
    y : series
        a series contraining all labels
    smote : object
        smote object for balancing the training data
    model : object
        logistic regression model
    allAttributes : list
        name of all attributes
    currentAttributes : list
        name of attributes currently used
    meanF1Score : float
        metric: F1 score
    AuRScore : float
        metric: area under ROC curve score
    cnfMatrix : list
        confusion matrix
    clasResport : dict
        dict containing the classification report
    ROC : list
        ROC curve
    PerbImp : list
        permutation importance of current attributes
    PerbImpMeanDict : dict
        dict containing permutation importance of current attributes


    Methods
    -------
    splitDataset()
        split the data set in training and test subsets
    update()
        train, test and eval model
    plot()
        plot feature importance and ROC vurve
    restrictAttribute()
        use only a subset of attributes
    evalPerformance()
        evaluate model performance
    createPerformanceReport()
        create a text file containing important performance metrics and feature importance
    plotFeatureImportance()
        plot feature impoirtance
    plotROC()
        plot ROC curve
    calcPerbImp()
        calculate permutation importance of attributes
    balanceDataset()
        balance training set with SMOTE
    fitModel()
        fit logistic regression model
    predictTestData()
        predict test data

    """
    def __init__(self, x, y,basefolder, testSplit= 0.2, seed = 0, PerbImp_repeats = 20):
        self.seed = seed
        self.basefolder = basefolder
        self.PerbImp_repeats = PerbImp_repeats
        self.model = LogisticRegression()
        self.smote = SMOTE(random_state=self.seed)
        self.testSplit = testSplit
        self.x = x
        self.y = y
        self.nameLable = y.name
        self.allAttributes = x.columns
        self.currentAttributes = self.allAttributes
        self.splitDataset()
        self.update()

    def splitDataset(self):
        self.trainX, self.testX, self.trainY, self.testY = \
            train_test_split(StandardScaler().fit_transform(self.x[self.currentAttributes]),
                             self.y, test_size=self.testSplit, random_state=self.seed)
        self.balanceDataset()

    def plot(self, folder):
        self.plotFeatureImportance(folder)
        self.plotROC(folder)

    def update(self):
        self.fitModel()
        self.predictTestData()
        self.calcPerbImp()
        self.evalPerformance()

    def restrictAttribute(self, Attributes):
        self.currentAttributes = Attributes
        self.splitDataset()
        self.update()

    def evalPerformance(self):
        self.meanF1Score= f1_score(self.testY, self.testY_pred, average="macro")
        self.AuRScore = roc_auc_score(self.testY, self.testY_predProb[:, 1])
        self.cnfMatrix = confusion_matrix(self.testY, self.testY_pred)
        self.clasResport = classification_report(self.testY, self.testY_pred, output_dict=True)
        self.ROC = roc_curve(self.testY, self.testY_predProb[:, 1])

    def createPerformanceReport(self, folder):
        path = os.path.join(self.basefolder,folder, "performanceReport.txt")
        file1 = open(path, "a")

        AttStr = "Used Attributes: "   + ('; '.join(str(x) for x in self.currentAttributes))
        file1.write(AttStr + "\n\n")

        file1.write('Area under ROC = %.3f' % (self.meanF1Score) + "\n")
        file1.write('Mean F1 Score = %.3f' % (self.AuRScore) + "\n\n")

        file1.write('Total Samples = %.3f' % (len(self.x)) + "\n")
        file1.write('Positive Samples = %.3f' % (len(self.y[self.y == 1])) + "\n")
        file1.write('Negative Samples = %.3f' % (len(self.y[self.y == 0])) + "\n\n")

        df = pd.DataFrame(self.clasResport).transpose()
        tab = tabulate(df, headers='keys', tablefmt='psql')
        file1.write(tab + "\n")

        file1.write("\n\n" + "Permutation Importance of Attributes" + "\n")
        df = pd.DataFrame(
            self.PerbImpMeanDict, index=["Permutation Importance"]
        ).T.sort_values(by=["Permutation Importance"],ascending=False)
        df["Permutation Importance Normalized"] = df["Permutation Importance"].apply(
            lambda x: x / df["Permutation Importance"].sum() * 100
        )
        df = df.round({'Permutation Importance': 4, "Permutation Importance Normalized": 1})
        tab = tabulate(df, headers='keys', tablefmt='psql')
        file1.write(tab + "\n")

        file1.write("\n\n" + "Confusion Matrix"  + "\n")
        df = pd.DataFrame(self.cnfMatrix, index=["0","1"])
        tab = tabulate(df, headers='keys', tablefmt='psql')
        file1.write(tab + "\n")
        file1.close()

    def plotFeatureImportance(self,folder = ""):
        fig = plt.figure()
        plt.bar(range(len(self.PerbImpMeanDict)), sorted(self.PerbImpMeanDict.values(), reverse=True), align='center')
        plt.xticks(range(len(self.PerbImpMeanDict)), sorted(self.PerbImpMeanDict, key=self.PerbImpMeanDict.get, reverse=True))
        plt.xticks(rotation=75)
        plt.ylabel('Permutation Feature Importance')
        path = os.path.join(self.basefolder, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        path =  os.path.join(path, 'permutation_importance.png')
        fig.savefig(path)

    def plotROC(self,folder = "", label = "ROC Curve"):
        path = os.path.join(self.basefolder, folder)
        plotSummaryROC([self.ROC], folder = path, labels=[label])

    def calcPerbImp(self):
        self.PerbImp = permutation_importance(
            self.model, self.testX, self.testY, scoring='neg_mean_squared_error',
            random_state=self.seed,n_repeats=self.PerbImp_repeats
        )
        self.PerbImpMean = self.PerbImp.importances_mean
        self.PerbImpMeanDict = dict(zip(self.currentAttributes, self.PerbImpMean.tolist()))

    def balanceDataset(self):
        trainX_res, trainY_res = self.smote.fit_resample(self.trainX, self.trainY)
        self.trainX_res = pd.DataFrame(trainX_res)
        self.trainY_res = pd.DataFrame(trainY_res)

    def fitModel(self):
        self.model.fit(self.trainX_res, self.trainY_res.values.ravel())

    def predictTestData(self):
        self.testY_pred = self.model.predict(self.testX)
        self.testY_predProb = self.model.predict_proba(self.testX)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputname', type=str, default=None, help="Name of data file", required=True)
    parser.add_argument('--NameLable', type=str, default="is_open", help="name of label", required=False)
    parser.add_argument('--folder', type=str, default="mount", help="Relative path of folder to operate in", required=False)
    return parser.parse_args()

def plotSummaryROC(ROCList, folder, labels):
    fig = plt.figure()
    line = np.linspace(0,1,11)
    plt.plot(line, line, linestyle='--', color = "black")
    color = reversed(sns.color_palette("GnBu", len(labels)+2))#"PuBu"
    for R, l, c in zip(ROCList, labels, color):
        plt.plot(R[0], R[1], marker= None, color = c, label=l, markersize=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, 'ROC.png')
    fig.savefig(path, bbox_inches='tight')

def main(args):
    folder = os.path.normpath(os.path.join(os.getcwd(), args.folder))
    path = os.path.join(folder, args.inputname)
    df = pickle.load(open(path, "rb"))
    string = "run_" + str(datetime.now()).replace(" ", "_").replace(".", "_").replace(":", "_").replace("'", "_")
    basefolder = os.path.join(folder, string)
    LogRegModel = Model(y = df[args.NameLable], x = df.drop([args.NameLable], axis=1), basefolder = basefolder)
    LogRegModel.plot(folder = "All_Attributes")
    LogRegModel.createPerformanceReport(folder = "All_Attributes")
    ROC = [LogRegModel.ROC]
    Labels = ["All_Attributes"]
    PerbImpDict = LogRegModel.PerbImpMeanDict
    for i in [10, 5, 3, 1]:
        LogRegModel.restrictAttribute(sorted(PerbImpDict, key=PerbImpDict.get, reverse=True)[:i])
        label = "Top_" + str(i) + "_Attribute"
        LogRegModel.plot(folder=label)
        Labels.append(label)
        ROC.append(LogRegModel.ROC)
    plotSummaryROC(ROCList = ROC, folder = basefolder, labels = Labels)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)