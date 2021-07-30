"""

This script reads in a dataframe containing the yelp business data and a dataframe containing the yelp review data and extracts features describing the busnesses.

Columns of input data:
Businesses: business_id, state, stars, review_count, is_open, hours, categories, latitude, longitude, city
Reviews: review_id, user_id, business_id, stars,date

Extracted features:
"stars", "review_count"                                                                         #   user rating
"latitude", "longitude"                                                                         #   location
"EveryDay_h", "total_h"                                                                         #   summary of opening hours
"Monday_h", "Tuesday_h", "Wednesday_h", "Thursday_h", "Friday_h", "Saturday_h", "Sunday_h"      #   detailed of opening hours
"ReviewsPerDay", "ratingTrend", "Age", "RewFreqTrend"                                           #   review features
"BusPerCity", "BusPerState"                                                                     #   business in the same location

"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class Yelp_Data:
    """
    A class contraining the yelp data

    ...

    Attributes
    ----------
    dfBus : dataframe
        a dataframe contraining the yelp business data
    dfRew : dataframe
        a dataframe contraining the yelp business data
    featuresWlabel : str
        a dataframe contraining the extracted features and the label
    RewGbBus : dict
        dict contraining reviews of a business with key = business_id

    Methods
    -------
    _GroupRewsByBus()
        create a dict conraining dataframes with the reviews of every business with key = business_id
    ratingAttributes()
        transfer rating attributes to feature dataframe
    locationAttributes()
        transfer location attributes to feature dataframe
    unpackHours()
        extracting features describing the opening hours
    ageAndFreqAttributes()
        extracting features describing age and rating frequency of businesses
    ratingTrendAttribute()
        calculate trend of user ratings
    rewFreqTrendAttribute(n = 10)
        calculate trend of rating frequency with n = number of intervals used
    competitionAttributes()
        calculating number of business in the same city/state for every business

    """
    def __init__(self, dfBus, dfRew):
        self.dfBus = dfBus.dropna()
        self.dfBus.set_index('business_id', inplace=True, drop=True)
        self.dfRew = dfRew.dropna()
        self.dfRew.set_index('review_id', inplace=True, drop=True)
        self.featuresWlabel = pd.DataFrame(index = self.dfBus.index)
        self.featuresWlabel = pd.concat([self.featuresWlabel, self.dfBus["is_open"]], axis=1)
        self.RewGbBus = None

    def _GroupRewsByBus(self):
        self.RewGbBus = dict(
            list(
                self.dfRew[self.dfRew["business_id"].isin(self.dfBus.index)].groupby(["business_id"])
            )
        )

    def ratingAttributes(self):
        self.featuresWlabel = pd.concat([self.featuresWlabel, self.dfBus[["stars", "review_count"]]], axis=1)#

    def locationAttributes(self):
        self.featuresWlabel = pd.concat([self.featuresWlabel, self.dfBus[["latitude", "longitude"]]], axis=1)#

    def unpackHours(self, how="complete"):
        df_hours = self.dfBus["hours"].apply(pd.Series)
        df_hours = df_hours.replace(np.nan, "0:0-0:0")
        df_hours = df_hours.applymap(
            lambda x: datetime.strptime(x.split("-")[1], "%H:%M") - datetime.strptime(x.split("-")[0], "%H:%M"))
        df_hours = df_hours.applymap(lambda x: (x.total_seconds() / 3600) % 24)
        if how in ["complete", "summary"]:
            df_hours = df_hours.assign(EveryDay=df_hours[df_hours.columns].eq(0).any(axis=1))
            df_hours = df_hours.assign(total=df_hours.apply(np.sum, axis=1))
        if how == "summary":
            df_hours = df_hours[["EveryDay", "total"]]
        df_hours.columns = [n + "_h" for n in df_hours.columns]
        self.featuresWlabel = pd.concat([self.featuresWlabel, df_hours], axis=1)

    def ageAndFreqAttributes(self):
        if self.RewGbBus is None:
            self._GroupRewsByBus()
        self.featuresWlabel["ReviewsPerDay"] = 0.0
        self.featuresWlabel["Age"] = 0.0
        for b in self.RewGbBus.keys():
            NumberReviews = len(self.RewGbBus[b])
            if NumberReviews < 2:
                continue
            deltaTime = (self.RewGbBus[b]["date"].max()-self.RewGbBus[b]["date"].min()).total_seconds() / (3600*24)
            self.featuresWlabel.at[b, "Age"] = deltaTime
            self.featuresWlabel.at[b, "ReviewsPerDay"] = NumberReviews / deltaTime

    def ratingTrendAttribute(self):
        if self.RewGbBus is None:
            self._GroupRewsByBus()
        self.featuresWlabel["ratingTrend"] = 0.0
        for b in self.RewGbBus.keys():
            NumberReviews = len(self.RewGbBus[b])
            if NumberReviews < 10:
                continue
            self.featuresWlabel.at[b, "ratingTrend"] = \
                LinearRegression().fit(self.RewGbBus[b]["date"].values.reshape(-1, 1),
                                       self.RewGbBus[b]["stars"].values.reshape(-1, 1)).coef_[0]

    def rewFreqTrendAttribute(self, numIntervals=10):
        if self.RewGbBus is None:
            self._GroupRewsByBus()
        self.featuresWlabel["RewFreqTrend"] = 0.0
        for b in self.RewGbBus.keys():
            NumberReviews = len(self.RewGbBus[b])
            if NumberReviews < 10:
                continue
            deltaTime = (self.RewGbBus[b]["date"].max() - self.RewGbBus[b]["date"].min()).total_seconds()
            startTime = self.RewGbBus[b]["date"].min()
            interval = deltaTime / numIntervals
            ilist = []
            for i in range(numIntervals):
                numbRewInInterval = len(self.RewGbBus[b][
                                            self.RewGbBus[b]["date"].between(
                                                startTime + timedelta(seconds=i * interval),
                                                startTime + timedelta(seconds=(i + 1) * interval)
                                            )
                                        ]
                                        )
                ilist.append(numbRewInInterval)
            self.featuresWlabel.at[b, "RewFreqTrend"] = \
                LinearRegression().fit(np.array(range(numIntervals)).reshape(-1, 1), np.array(ilist).reshape(-1, 1)).coef_[0]

    def competitionAttributes(self):
        self.featuresWlabel["BusPerState"] = list(self.dfBus.groupby(["state"]).size()[self.dfBus["state"][self.featuresWlabel.index]])
        self.featuresWlabel["BusPerCity"] = list(self.dfBus.groupby(["city"]).size()[self.dfBus["city"][self.featuresWlabel.index]])

def main(args):
    folder = os.path.normpath(os.path.join(os.getcwd(), args.folder))
    path = os.path.join(folder, args.inputnameBusinesses)
    dfBus = pickle.load(open(path, "rb"))
    path = os.path.join(folder, args.inputnameReviews)
    dfRew = pickle.load(open(path, "rb"))
    data = Yelp_Data(dfBus = dfBus, dfRew= dfRew)
    if args.competitionAttributes:
        data.competitionAttributes()
    if args.ratingAttributes:
        data.ratingAttributes()
    if args.locationAttributes:
        data.locationAttributes()
    if args.obeningHourAttributes:
        data.unpackHours()
    if args.ageAndFreqAttributes:
        data.ageAndFreqAttributes()
    if args.ratingTrendAttribute:
        data.ratingTrendAttribute()
    if args.rewFreqTrendAttribute:
        data.rewFreqTrendAttribute(numIntervals=args.numIntervals)
    path = os.path.join(folder, args.outputname)
    pickle.dump(data.featuresWlabel, open(path, "wb"))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputnameBusinesses', type=str, default=None, help="Name of the input business dataframe", required=True)
    parser.add_argument('--inputnameReviews', type=str, default=None, help="Name of the input review datafram", required=True)
    parser.add_argument('--outputname', type=str, default="FeaturesWithLabels", help="Dataframe containing the features and labels is saved under this name", required=False)
    parser.add_argument('--numIntervals', type=int, default=10, help="Number of time intervals used for calculating the rating frequency trend", required=False)
    parser.add_argument('--obeningHourAttributes', type=int, default=1, help="extract features describing the opening hours or not", required=False)
    parser.add_argument('--competitionAttributes', type=int, default=1, help="extract features describing the competition or not", required=False)
    parser.add_argument('--ratingAttributes', type=int, default=1, help="extract features describing the rating or not", required=False)
    parser.add_argument('--locationAttributes', type=int, default=1, help="extract features describing the location or not", required=False)
    parser.add_argument('--ageAndFreqAttributes', type=int, default=1, help="extract features describing the age and review frquency or not", required=False)
    parser.add_argument('--ratingTrendAttribute', type=int, default=1, help="extract features describing the rating trend or not", required=False)
    parser.add_argument('--rewFreqTrendAttribute', type=int, default=1, help="extract features describing the the review frequency trend or not", required=False)
    parser.add_argument('--folder', type=str, default="mount", help="Relative path of folder to operate in", required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)