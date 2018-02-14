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
import random

import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

ACTIVATION_FUNCTIONS = ["RELU", "SIGMOID", "TANH", "SOFTSIGN", "ELU"]
NEURAL_NETWORK = "NeuralNetwork"
NEURAL_NETWORK_TEMPLATE = {
            "model": NEURAL_NETWORK,
            "maxLayersCount": 5,
            "maxLayerNeuronsCount": 300,
            "activationFunctions": ACTIVATION_FUNCTIONS
        }

class NeuralNetwork(object):
    def __init__(self, parameters=None, layersCount=0, deviceCount={'CPU' : 8, 'GPU' : 0}):

        tf.reset_default_graph()

        self.validActivationFunctions = [
            tf.nn.relu,
            tf.nn.sigmoid,
            tf.nn.tanh,
            tf.nn.softsign,
            tf.nn.elu]

        # Network Parameters
        self.layerDimensions = [x for x in parameters[:layersCount] if x > 0]
        self.layersCount = layersCount
        self.maxHiddenLayersCount = layersCount-2
        # Input layer without activation functions
        self.activationFunctions = [x for x in parameters[layersCount:]]

        # Parameters
        self.learningRate = 0.01
        self.displayStep = 1000

        # Store layers weights
        self.weights = []
        prevSize = self.layerDimensions[0]
        for i in range(1, len(self.layerDimensions)):
            self.weights.append(tf.Variable(tf.truncated_normal([prevSize, self.layerDimensions[i]], stddev=0.1, dtype=tf.float64), dtype=tf.float64))
            prevSize = self.layerDimensions[i]

        # Store layers biases
        self.biases = []
        for i in range(1, len(self.layerDimensions)):
            self.biases.append(tf.Variable(tf.constant(0.1, shape=[self.layerDimensions[i]], dtype=tf.float64), dtype=tf.float64))

        self.x_data = tf.placeholder(tf.float64, [None, self.layerDimensions[0]])
        self.y_data = tf.placeholder(tf.float64, [None, self.layerDimensions[-1]])

        # Construct model
        self.pred = self.buildMultilayerPerceptron()

        # Minimize the mean squared errors.
        self.loss = tf.reduce_mean(tf.square(self.pred - self.y_data))
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # Before starting, initialize the variables.  We will 'run' this first.
        self.init = tf.global_variables_initializer()

        # Launch the graph.
        config = tf.ConfigProto(
            intra_op_parallelism_threads=8, 
            inter_op_parallelism_threads=8, 
            allow_soft_placement=True, 
            device_count = deviceCount)
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def buildMultilayerPerceptron(self):
        prevLayer = self.x_data
        for i in range(len(self.layerDimensions)-1): 
            # Input layer without activation functions
            layer = self.activation(tf.add(tf.matmul(prevLayer, self.weights[i]), self.biases[i]), i)
            prevLayer = layer
        return prevLayer

    def activation(self, x, layerNumber):
        value = None
        for i in range(x.get_shape()[1].value):
            if len(self.activationFunctions)>0:
                activationFunction = self.validActivationFunctions[self.activationFunctions[layerNumber*self.maxHiddenLayersCount+i]]
            else:
                activationFunction = self.validActivationFunctions[0]
            if value is None:
                value = [activationFunction(x[:, i])] 
            else: 
                value = tf.concat(axis=0, values=[value, [activationFunction(x[:, i])]])
        return tf.transpose(value)

    def run(self, x):
        try:
            y = self.sess.run(self.pred, feed_dict={self.x_data: x})
            return y
        except:
            return None

    def load(self, loadPath):
        with gfile.FastGFile(os.path.join(loadPath, "model.pb"),'rb') as f:
            graphDef = tf.GraphDef()
            text_format.Merge(f.read().decode(), graphDef)
            self.sess.graph.as_default()
            tf.import_graph_def(graphDef, name='')
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, os.path.join(loadPath, "checkpoint.data"))

    def close(self):
        self.sess.close()
        return 1

