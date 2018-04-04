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

class Data(object):

    DATA_URL = "http://fremont1.cto911.com/esdoc/data/"

    def __init__(self, inputTrainCsvZipUrl, outputTrainCsvZipUrl, inputTestCsvZipUrl, outputTestCsvZipUrl, zipPassword=None, refreshData=True):
        self.inputTrainCsvZipUrl = inputTrainCsvZipUrl
        self.outputTrainCsvZipUrl = outputTrainCsvZipUrl
        self.inputTestCsvZipUrl = inputTestCsvZipUrl
        self.outputTestCsvZipUrl = outputTestCsvZipUrl
        self.zipPassword = zipPassword
        self.refreshData = refreshData

def load_valuation(refresh_data=True):
    inputTrainCsvZipUrl = '{}valuation_train_inputs.zip'.format(Data.DATA_URL)
    outputTrainCsvZipUrl = '{}valuation_train_outputs.zip'.format(Data.DATA_URL)
    inputTestCsvZipUrl = None
    outputTestCsvZipUrl = None    
    return Data(inputTrainCsvZipUrl, outputTrainCsvZipUrl, inputTestCsvZipUrl, outputTestCsvZipUrl, refresh_data)

def load_iris(refresh_data=True):
    inputTrainCsvZipUrl = '{}iris_train_inputs.zip'.format(Data.DATA_URL)
    outputTrainCsvZipUrl = '{}iris_train_outputs.zip'.format(Data.DATA_URL)
    inputTestCsvZipUrl = None
    outputTestCsvZipUrl = None
    return Data(inputTrainCsvZipUrl, outputTrainCsvZipUrl, inputTestCsvZipUrl, outputTestCsvZipUrl, refresh_data)


def load_diabetes(refresh_data=True):
    inputTrainCsvZipUrl = '{}diabetes_train_inputs.zip'.format(Data.DATA_URL)
    outputTrainCsvZipUrl = '{}diabetes_train_outputs.zip'.format(Data.DATA_URL)
    inputTestCsvZipUrl = None
    outputTestCsvZipUrl = None
    return Data(inputTrainCsvZipUrl, outputTrainCsvZipUrl, inputTestCsvZipUrl, outputTestCsvZipUrl, refresh_data)


def load_mnist(refresh_data=True):
    inputTrainCsvZipUrl = '{}mnist_train_inputs.zip'.format(Data.DATA_URL)
    outputTrainCsvZipUrl = '{}mnist_train_outputs.zip'.format(Data.DATA_URL)
    inputTestCsvZipUrl = '{}mnist_test_inputs.zip'.format(Data.DATA_URL)
    outputTestCsvZipUrl = '{}mnist_test_outputs.zip'.format(Data.DATA_URL)
    return Data(inputTrainCsvZipUrl, outputTrainCsvZipUrl, inputTestCsvZipUrl, outputTestCsvZipUrl, refresh_data)
