"""

This script reads in .json files and saves the data in a pandas dataframe.

"""
import pandas as pd
import numpy as np
import os
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None, help="Name of the json file to read", required=True)
    parser.add_argument('--outputname', type=str, default=None, help="Dataframe is saved under this name", required=False)
    parser.add_argument('--columns', type=str, default=None, help="Read only some columns (speperator = ,)", required=False)
    parser.add_argument('--query', type=str, default=None, help="filter data with query", required=False)
    parser.add_argument('--chunksize', type=int, default=2000, help="Chunksize for reading the data", required=False)
    parser.add_argument('--folder', type=str, default="mount", help="Relative path of folder to operate in", required=False)
    return parser.parse_args()

def readData(fileName, cols = None, query = None, chunksize=4000):
    b_pandas = []
    with open(fileName + ".json", "r", errors='replace') as f:
        if cols is not None:
            reader = pd.read_json(f, orient="records", lines=True, chunksize=chunksize)
        else:
            reader = pd.read_json(f, orient="records", lines=True, chunksize=chunksize)
        for chunk in reader:
            if cols is not None:
                chunk = chunk[cols]
            if query is not None:
                chunk = chunk.query(query)
            b_pandas.append(chunk)
    df = pd.concat(b_pandas, ignore_index=True)
    return df

def main(args):
    columns = None
    if args.columns is not None:
        columns = args.columns.replace(' ', '').split(',')
    folder = os.path.normpath(os.path.join(os.getcwd(), args.folder))
    path = os.path.join(folder, args.filename)
    df = readData(path, cols=columns,chunksize = args.chunksize, query=args.query)#, r_dtypes=r_dtypes
    outputname = args.filename if args.outputname is None else args.outputname
    path = os.path.join(folder, outputname)
    pickle.dump(df, open(path, "wb"))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)