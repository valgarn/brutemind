#
#   MIT License
#
#    brutemind framework for python
#    Copyright (C) 2018 Michael Lin, Valeriy Garnaga
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import io
import sys
from enum import Enum
import requests
import json
import base64
import zipfile
import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

import brutemind.NeuralNetwork as nn

class API(object):

    SERVER = "http://oliver.dnsdojo.org:5001"
    #SERVER = "http://127.0.0.1:5001"
    TRANSFER_DATA_ROWS_PER_REQUEST = 1000
    MODELS = ["NeuralNetwork"]
    ACTIVATION_FUNCTIONS = ["RELU", "SIGMOID", "TANH", "SOFTSIGN", "ELU"]
    NEURAL_NETWORK_TEMPLATE = {
                "model": "NeuralNetwork",
                "maxLayersCount": 5,
                "maxLayerNeuronsCount": 300,
                "activationFunctions": ACTIVATION_FUNCTIONS
            }

    def __init__(self, modelsList=None, inputData=None, outputData=None, autentificationToken=None, \
                        transferDataRowsPerRequest=None, server=None, deviceCount={'CPU' : 8, 'GPU' : 0}, \
                        intraOpParallelismThreads=8, interOpParallelismThreads=8, allowSoftPlacement=True):
        self.modelsList = modelsList if not modelsList is None else []
        self.inputData = inputData if not inputData is None else []
        self.outputData = outputData if not outputData is None else []
        self.autentificationToken = autentificationToken
        self.transferDataRowsPerRequest = transferDataRowsPerRequest if not transferDataRowsPerRequest is None else API.TRANSFER_DATA_ROWS_PER_REQUEST
        self.server = server if not server is None else API.SERVER
        self.deviceCount = deviceCount
        self.intraOpParallelismThreads = intraOpParallelismThreads
        self.interOpParallelismThreads = interOpParallelismThreads
        self.allowSoftPlacement = allowSoftPlacement
        self.model = None

    def sendModelsList(self):
        response = requests.post("{}/api/v1/register_models_list".format(self.server), 
            data=json.dumps(
                {
                    "transferDataRowsPerRequest": self.transferDataRowsPerRequest,
                    "autentificationToken": self.autentificationToken,
                    "modelsList": self.modelsList
                }))
        return response.json()

    def sendData(self):
        n = len(self.inputData)
        if n!=len(self.outputData):
            raise RuntimeError("Error: Unequal sizes of input and output data.")
        else:
            for i in range(0, n, self.transferDataRowsPerRequest):
                j = (i+self.transferDataRowsPerRequest if n>=i+self.transferDataRowsPerRequest else n)
                print(i, j)
                response = requests.post("{}/api/v1/add_data".format(self.server), 
                    data=json.dumps(
                        {
                            "autentificationToken": self.autentificationToken,
                            "inputData": self.inputData[i:j],
                            "outputData": self.outputData[i:j]
                        }))
                return response.json()

    def start(self):
        self.sendModelsList()
        self.sendData()
        response = requests.post("{}/api/v1/start".format(self.server), 
            data=json.dumps(
                {
                    "autentificationToken": self.autentificationToken
                }))
        return response.json()

    def getStatus(self):
        response = requests.post("{}/api/v1/get_status".format(self.server), 
            data=json.dumps(
                {
                    "autentificationToken": self.autentificationToken
                }))
        return response.json()

    def getBestModel(self, path):
        response = requests.post("{}/api/v1/get_best_model".format(self.server),
            data=json.dumps(
                {
                    "autentificationToken": self.autentificationToken
                }))

        args = response.json()

        id = args.get('id', None)
        if id is None:
            return None, None, None
        
        loss = args['loss']
        folder = os.path.join(path, id)
        chromosome = args['chromosome']
        layersCount = args['layersCount']

        model = io.BytesIO(base64.b64decode(args['modelZip']))
        zpf = zipfile.ZipFile(model, "r")
        if not os.path.exists(folder):
            os.mkdir(folder)
        zpf.extractall(folder)
        zpf.close()

        self.model = nn.NeuralNetwork(chromosome, layersCount, {'CPU' : 8, 'GPU' : 0})
        self.model.load(os.path.join(folder))
        return id, loss, self.model

    def stop(self):
        response = requests.post("{}/api/v1/stop".format(self.server), 
            data=json.dumps(
                {
                    "autentificationToken": self.autentificationToken
                }))
        return response.json()

    def clear(self):
        response = requests.post("{}/api/v1/clear".format(self.server), 
            data=json.dumps(
                {
                    "autentificationToken": self.autentificationToken
                }))
        return response.json()

