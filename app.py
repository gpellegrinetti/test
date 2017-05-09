import pandas as pd
from pandas import DataFrame
import random as rd
import threading
import xgboost as xgb

import json
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score

from flask import Flask, request

app = Flask(__name__)

@app.route('/hello')
def hello():
    return '{"Hello": "World"}'

@app.route('/readInput')
def readInput():
    file = request.args.get('file')
    dt = pd.read_csv(file)
    return dt.to_json()

@app.route('/createInput', methods=['PUT'])
def createInput():
    print(request.query_string)
    nrow = int(request.args.get('nrow'))
    file = request.args.get('file')

    dt = pd.read_csv("iris.csv", nrows=nrow)
    dt["length"] = dt["petal_length"] + rd.uniform(0, 0.5)
    dt["width"] = dt["petal_width"] + rd.uniform(0, 0.1)
    dt.to_csv(file)
    return '{"result": "ok"}'


@app.route('/trainModel')
def trainModel():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # use DMatrix for xgbosot
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # use svmlight file for xgboost
    #dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
    #dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
    #dtrain_svm = xgb.DMatrix('dtrain.svm')
    #dtest_svm = xgb.DMatrix('dtest.svm')

    # set xgboost params
    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 3}  # the number of classes that exist in this datset
    num_round = 20  # the number of training iterations

    #------------- numpy array ------------------
    # training and testing - numpy matrices
    global bst
    bst = xgb.train(param, dtrain, num_round)
    return '{"result": "ok"}'

@app.route('/predictModel', methods=['POST'])
def predictModel():
    dati = request.form.get('dati')
    datiJS = json.loads(dati)
    df = pd.DataFrame.from_dict(datiJS)
    dl = df.as_matrix()
    dtest = xgb.DMatrix(dl)
    preds = bst.predict(dtest)
    predsL = preds.tolist()
    return json.dumps(predsL)




if __name__ == "__main__":
    app.run()
