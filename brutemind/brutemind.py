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
import pathlib
from enum import Enum
import requests
import json
import base64
import zipfile
import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import threading
import time
import shutil
from brutemind import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

class GeneticAlgoSearch(object):

    SERVER = "http://oliver.dnsdojo.org:5001"
    TRANSFER_DATA_ROWS_PER_REQUEST = 1000
    FOLDER = "models"

    def __init__(self, parameters=None, autentificationToken=None, \
                        transferDataRowsPerRequest=None, server=None, deviceCount=None, \
                        intraOpParallelismThreads=8, interOpParallelismThreads=8, allowSoftPlacement=True):
        self.parameters = parameters        
        self.autentificationToken = autentificationToken
        self.transferDataRowsPerRequest = transferDataRowsPerRequest if not transferDataRowsPerRequest is None \
                                                                        else GeneticAlgoSearch.TRANSFER_DATA_ROWS_PER_REQUEST
        self.server = server if not server is None else GeneticAlgoSearch.SERVER
        self.deviceCount = deviceCount if not deviceCount is None else {'CPU' : 8, 'GPU' : 0}
        self.intraOpParallelismThreads = intraOpParallelismThreads
        self.interOpParallelismThreads = interOpParallelismThreads
        self.allowSoftPlacement = allowSoftPlacement
        self.model = None
        self.stopPredictFlag = True

    def get_params(self):
        return {
            "status": self.getStatus(),
            "params": self.parameters
        }

    def get_result(self):
        return self.getBestModel(GeneticAlgoSearch.FOLDER)

    def predictResult(self, data=None):
        accuracy = None
        output = []
        id, loss, model, folder, _ = self.getBestModel(GeneticAlgoSearch.FOLDER)
        if not id is None:
            correctAnswersCount = 0
            for i in range(len(self.inputData)):
                y = model.run([self.inputData[i]])
                if y is None:
                    return None, None, None
                else:
                    n = len(self.outputData[i])
                    if n==1:
                        if (self.outputData[i][0] == 1.0 and y[0][0]>0.5) or (self.outputData[i][0] == 0.0 and y[0][0]<=0.5):
                            correctAnswersCount += 1
                    else:
                        okIndex = 0
                        resultIndex = 0
                        resultValue = y[0][0]                    
                        for j in range(1, n):
                            if self.outputData[i][j] == 1.0:
                                okIndex = j
                            if resultValue < y[0][j]:
                                resultIndex = j
                                resultValue = y[0][j]                        
                        if okIndex==resultIndex:
                            correctAnswersCount += 1
            accuracy = 100.0 * correctAnswersCount / len(self.outputData)
            if not data is None:
                for i in range(len(data)):
                    output.append(model.run([data[i]]))
        return accuracy, folder, output

    def predictThread(self, data=None, callback=None, accuracy=None, stop=True):
        maxAccuracy = 0
        while not self.stopPredictFlag:
            a, f, o = self.predictResult(data)
            if not a is None and a > maxAccuracy:
                maxAccuracy = a
                if (not accuracy is None and a>=accuracy) or accuracy is None:
                    callback(self, a, f, o)
                    if stop:
                        self.stopPredictFlag = True
                else:
                    if not f is None:
                        shutil.rmtree(f, ignore_errors=True)
            else:
                if not f is None:
                    shutil.rmtree(f, ignore_errors=True)
            time.sleep(3)

    def predict(self, data=None):
        return self.predictResult(data)

    def fit(self, inputData, outputData, callback=None, accuracy=None, stop=True):
        self.clear()
        self.inputData = inputData
        self.outputData = outputData
        self.start()
        if not callback is None:
            self.stopPredictFlag = False
            callbackThread = threading.Thread(target=self.predictThread, args=(None, callback, accuracy, stop))
            callbackThread.start()

    def sendModelsList(self):
        response = requests.post("{}/api/v1/register_models_list".format(self.server), 
            data=json.dumps(
                {
                    "transferDataRowsPerRequest": self.transferDataRowsPerRequest,
                    "autentificationToken": self.autentificationToken,
                    "modelsList": self.parameters
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
        try:
            response = requests.post("{}/api/v1/get_best_model".format(self.server),
                data=json.dumps(
                    {
                        "autentificationToken": self.autentificationToken
                    }))
            args = response.json()
            id = args.get('id', None)
            if id is None:
                return None, None, None, None, "Processing currently ..."
            loss = args['loss']
            folder = os.path.join(path, "{} {}".format(time.strftime("%Y %b %d %H:%M:%S"), id))
            chromosome = args['chromosome']
            layersCount = args['layersCount']
            model = io.BytesIO(base64.b64decode(args['modelZip']))
            zpf = zipfile.ZipFile(model, "r")
            if not os.path.exists(folder):
                pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
            zpf.extractall(folder)
            zpf.close()
            self.model = models.NeuralNetwork(chromosome, layersCount, {'CPU' : 8, 'GPU' : 0})
            self.model.load(os.path.join(folder))
            return id, loss, self.model, folder, "Working ..."
        except:
            return None, None, None, None, "Waiting ..."

    def stop(self):
        self.stopPredictFlag = True
        response = requests.post("{}/api/v1/stop".format(self.server), 
            data=json.dumps(
                {
                    "autentificationToken": self.autentificationToken
                }))
        return response.json()

    def clear(self):
        self.stop()
        tryCount = 10
        while tryCount>0:
            tryCount -= 1
            try:
                if os.path.exists(GeneticAlgoSearch.FOLDER):          
                    shutil.rmtree(GeneticAlgoSearch.FOLDER)
                    os.mkdir(GeneticAlgoSearch.FOLDER)                
            except:
                time.sleep(5)
        response = requests.post("{}/api/v1/clear".format(self.server), 
            data=json.dumps(
                {
                    "autentificationToken": self.autentificationToken
                }))
        return response.json()

