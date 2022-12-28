from flask import Flask, request
from flask_restful import Resource,Api
from flask_cors import CORS
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import json
from ID3 import ID3

# Inisiasi object flask
app = Flask(__name__)

# Database Configuration
app.config['MYSQL_HOST'] = 'sbp.ridlwany.com'
app.config['MYSQL_USER'] = 'sbp'
app.config['MYSQL_PASSWORD'] = 'sbp'
app.config['MYSQL_DB'] = 'sbpdb'

# Inisiasi object flask_restful
api = Api(app)
mysql = MySQL(app)

# Insiasi object flask_cors
CORS(app)

identitas = {}

class ContohResource(Resource):
    # metode get dan post
    def get(self):
        curDataset = mysql.connection.cursor()
        curDataset.execute("SELECT d.* FROM dataset d")
        dataset = curDataset.fetchall()
        curDataset.close()

        curFeature = mysql.connection.cursor()
        curFeature.execute("SELECT f.feature_name FROM feature f")
        feature = curFeature.fetchall()
        curFeature.close()

        datasets = [list(item[1:len(feature)+1]) for item in dataset]
        features = [item[0] for item in feature]

        train_data_m = pd.DataFrame(data=datasets, columns=features)
        id3 = ID3(train_data_m, 'idaman')
        tree = id3.decision_tree()

        return tree

    def post(self):
        message = request.form["message"]

        curDataset = mysql.connection.cursor()
        curDataset.execute("SELECT d.* FROM dataset d")
        dataset = curDataset.fetchall()
        curDataset.close()

        curFeature = mysql.connection.cursor()
        curFeature.execute("SELECT f.feature_name FROM feature f")
        feature = curFeature.fetchall()
        curFeature.close()

        datasets = [list(item[1:len(feature) + 1]) for item in dataset]
        features = [item[0] for item in feature]

        train_data_m = pd.DataFrame(data=datasets, columns=features)
        id3 = ID3(train_data_m, 'idaman')
        tree = id3.decision_tree()

        item = json.loads(message)
        json_feature = []

        i = 0
        for feature_name in features:
            # datatest = [a[0] for a in item[feature_name]]
            json_feature.insert(i,item[feature_name])
            i = i+1
            print(item[feature_name])

        print([list(json_feature)])
        datatest = np.array([list(json_feature)])
        test_data_m = pd.DataFrame(data=datatest, columns=features)
        result = id3.predict(tree, test_data_m.iloc[0])
        #response = datatest
        #print(datatest)
        print(tree)
        return result

# Setup Resource
api.add_resource(ContohResource, "/api", methods=["GET", "POST"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)