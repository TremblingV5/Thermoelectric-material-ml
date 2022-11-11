import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier

from misc import get_train_data, validate

XTrain, XTest, YTrain, YTest = get_train_data()

# 各个方法的参数以键值对的形式写在对应算法的args参数之后
methods = {
    "随机森林": {
        "class": RandomForestClassifier,
        "args": {}
    },
    "SVM-LinearSVC": {
        "class": LinearSVC,
        "args": {}
    },
    "K近邻": {
        "class": KNeighborsClassifier,
        "args": {}
    },
    "Naive Bayes": {
        "class": MultinomialNB,
        "args": {}
    },
    "随机梯度下降": {
        "class": SGDClassifier,
        "args": {}
    },
    "多层感知器": {
        "class": MLPClassifier,
        "args": {}
    },
    "逻辑回归": {
        "class": LogisticRegression,
        "args": {}
    },
    "AdaBoost": {
        "class": AdaBoostClassifier,
        "args": {}
    },
    "梯度提升": {
        "class": GradientBoostingClassifier,
        "args": {}
    }
}

for item in methods:
    print("算法：", item)
    if methods[item]["args"] == {}:
        model = methods[item]["class"]()
    else:
        model = methods[item]["class"](**methods[item]["args"])
    model.fit(XTrain, YTrain)
    validate(model.predict(XTest), YTest)
