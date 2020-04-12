# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:12:51 2020

@author: Rishabh
"""

from flask import Flask, jsonify, request, Response
import json
import pickle
import numpy as np

with open('rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)


def valid_iris(iris_object):
    if ("sepal_length" in iris_object and "sepal_width" in iris_object and "petal_length" in iris_object and "petal_width" in iris_object):
        return True
    else:
        return False

#POST /books
@app.route('/classify', methods=['POST'])
def iris_classification():
    request_data = request.get_json()
    '''
    print(request_data)
    print('******************')
    print(request_data['sepal_length'])
    print('******************')
    print(request_data['sepal_width'])
    print('******************')
    print(request_data['petal_length'])
    print('******************')
    print(request_data['petal_width'])
    print('******************')
    print("sepal_length" in request_data)
    '''
    
    
    if(valid_iris(request_data)):
        s_length = request_data['sepal_length']
        s_width = request_data['sepal_width']
        p_length = request_data['petal_length']
        p_width = request_data['petal_width']
        prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
        print(type(prediction))
        print(prediction[0])
#        return str(prediction)
        result = int(prediction[0])
        print(type(result))
        print(result)
        if result == 0:
            output = "setosa "
        elif result == 1:
            output = "versicolor"
        elif output == 2:
            output = "virginica"
        return jsonify({"class": output})
    else:
        invalidIrisObjectErrorMsg = {
            "error": "Invalid iris object passed in request",
            "helpString": "Data passed in similar to this {'sepal_length': 5, 'sepal_width': 2.9, 'petal_length': 1, 'petal_width': 0.2 }"
        }
        response = Response(json.dumps(invalidIrisObjectErrorMsg), status=400, mimetype='application/json')
        return response;

app.run(port=5000)
