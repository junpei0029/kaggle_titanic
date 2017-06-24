#!/usr/bin/env python

from __future__ import print_function

import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE

# stacking
from stacked_generalization.lib.stacking import StackedClassifier

# 分類器
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import xgboost as xgb


def get_titanic():
    train_data = pd.read_csv("datasets/train.csv", header="infer")
    df = pd.DataFrame(train_data)
    df = df[['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    df = pd.get_dummies(df[['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
    
    # l = [0.05, 0.25, 0.5, 0.75, 0.99]
    # num_over_99 = df.describe(percentiles=l)['Fare']['99%']
    # df["Fare"].where(df["Fare"] > num_over_99, None).dropna()

    imr = Imputer(missing_values='NaN', strategy='median', axis=0)
    imr = imr.fit(df)
    train = imr.transform(df.values)

    # 標準化
    # sc = StandardScaler()
    # train = sc.fit_transform(train)

    X = train[:,2:]
    y = train[:,1]
    return X, y

def get_titanic_test():
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
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    x_train, y_train = get_titanic()

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    n_trees = 10
    # Our level 0 classifiers
    clfs = [
        RandomForestClassifier(n_estimators = n_trees, criterion = 'gini'),
        ExtraTreesClassifier(n_estimators = n_trees * 2, criterion = 'gini'),
        GradientBoostingClassifier(n_estimators = n_trees),
    ]


    clf1 = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=3)
    clf2 = RandomForestClassifier(criterion='entropy', random_state=0)
    clf3 = LogisticRegression(penalty='l2', random_state=0)
    param_grid_clf3 = [{'C': [0.001,0.01,0.1, 1.0,10.0,100.0,1000.0]}]
    clf4 = TSNE(n_components=2, perplexity=50, verbose=3, random_state=1)
    clf5 = xgb.XGBClassifier(verbose=1)
    param_grid_clf5 = [{'max_depth': [2,4,6], 'n_estimators': [50,100,200]}]

    pipe_svc = Pipeline([('scl', StandardScaler()), 
                        ('clf', SVC(random_state=1))
                        ]) 
    param_range = [0.001,0.01,0.1, 1.0,10.0,100.0,1000.0]
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

    ada = AdaBoostClassifier(base_estimator=clf1,
                             n_estimators=500,
                             random_state=1
        )
    param_grid_ada = [{'learning_rate': [0.1, 0.2, 0.3]}]

    # pipe_tree = Pipeline([('tree', DecisionTreeClassifier(criterion='entropy', random_state=0))])
    # param_grit_tree = [{'tree__max_depth': [1,2,3,4,5,6,7,None]}]

    # pipe_rnfr = Pipeline([('rnfr', RandomForestClassifier(criterion='entropy', random_state=0))])
    # param_grit_rnfr = [{'rnfr__n_estimators': [5,10,15,20,25,30],'rnfr__max_depth': [1,2,3,4,5,6,7,None]}]

    cv = KFold(n_splits=10, shuffle=True)
    gs = GridSearchCV(estimator=clf5,
                    param_grid=param_grid_clf5,
                    scoring='accuracy',
                    cv=cv,
                    n_jobs=1
                    )
    # scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=5)
    # print(np.mean(scores),np.std(scores))

    gs.fit(x_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    x_test = get_titanic_test()
    x_test_pre = x_test[:,1:]

    # 予測
    model = gs.best_estimator_
    label_prediction = model.predict(x_test_pre)

    # CSV出力
    submission = pd.DataFrame()
    submission['PassengerId'] = x_test[:,0].astype(np.int64)
    submission['Survived'] = label_prediction.astype(np.int64)
    submission.to_csv("predict.csv", index=False)

if __name__ == '__main__':
    main()
