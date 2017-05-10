import pandas as pd
from pandas import DataFrame
import random as rd

from flask import Flask, request

app = Flask(__name__)

@app.route('/hello')
def hello():
    return '{"Hello": "Nives"}'

@app.route('/readInput')
def readInput():
    file = request.args.get('file')
    dt = pd.read_csv(file)
    return dt.to_json()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
