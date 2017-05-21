#!/usr/bin/env python

from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Variable
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import nets

def predict(model, x_data):
    x = Variable(x_data.astype(np.float32))
    y = model.predictor(x)
    return np.argmax(y.data, axis=1)

def get_test_titanic():
    data = pd.read_csv("datasets/test.csv", header="infer")
    df = pd.DataFrame(data)
    df = df[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    df = pd.get_dummies(df[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
    imr = Imputer(missing_values='NaN', strategy='median', axis=0)
    imr = imr.fit(df)
    test = imr.transform(df.values)
    return test

def main():
    parser = argparse.ArgumentParser(description='Chainer example: Titanic')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('# unit: {}'.format(args.unit))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(nets.MLP(args.unit, 2, False, 0.5))

    test_df = get_test_titanic()
    x_test = test_df[:,1:]
    # 標準化
    sc = StandardScaler()
    x_test = sc.fit_transform(x_test)

    chainer.serializers.load_npz("train_chainer.model", model)
    label_prediction = predict(model, x_test)

    print(x_test.shape)
    print(x_test.shape[0])

    ids = test_df[:,0].reshape(x_test.shape[0],1)
    result = np.hstack((ids,label_prediction.reshape(x_test.shape[0],1)))
    np.savetxt("predict_chainer.csv", result, delimiter=",", header="PassengerId,Survived", fmt='%.0f')

if __name__ == '__main__':
    main()
