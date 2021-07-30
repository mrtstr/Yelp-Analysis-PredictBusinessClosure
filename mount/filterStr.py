"""

This script reads in a dataframe and filters the rows based on substrings in a specific column and saves the filtered dataframe.
Only include rows witch contain at least one string of the strInc's as a substring in a specific column with the name columnName.
Exclude strings witch contain at least one string of the strExc's as a substring in a specific column with the name columnName.
Filter some chars.

"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputname', type=str, default=None, help="Name of the input dataframe", required=True)
    parser.add_argument('--strInc', type=str, default=None, help="Only include rows witch contain at least one string of the strInc's as a substring in a specific column with the name columnName (seperator = ,)", required=True)
    parser.add_argument('--columnName', type=str, default="categories", help="Filter based on the strings in this column", required=False)
    parser.add_argument('--outputname', type=str, default=None, help="Dataframe is saved under this name", required=False)
    parser.add_argument('--strExc', type=str, default=None, help="Exclude strings witch contain at least one string of the strExc's as a substring in a specific column with the name columnName (seperator = ,)", required=False)
    parser.add_argument('--folder', type=str, default="mount", help="Relative path of folder to operate in", required=False)
    return parser.parse_args()

def main(args):
    folder = os.path.normpath(os.path.join(os.getcwd(), args.folder))
    path = os.path.join(folder, args.inputname)
    df = pickle.load(open(path, "rb"))
    strInc = args.strInc.split(',')
    columnName = args.columnName
    df[columnName] = df[columnName].apply(lambda x: str(x).replace(" ", "").replace("&", "").replace("(", "").replace(")", ""))
    df = df[df[columnName].notna()]
    mask = df[columnName].apply(lambda x: any([c in x for c in strInc]))
    df = df[mask]
    if args.strExc is not None:
        strExc = args.strExc.split(',')
        mask = df[columnName].apply(lambda x: not any([c in x for c in strExc]))
        df = df[mask]
    outputname = args.inputame if args.outputname is None else args.outputname
    path = os.path.join(folder, outputname)
    pickle.dump(df, open(path, "wb"))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)