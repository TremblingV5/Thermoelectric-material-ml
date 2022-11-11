import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import model_selection

def get_train_data():
    df = pd.read_csv("./ucsb_test.csv")
    X = df[[
        "MagpieData mean MeltingT",
        "MagpieData mean NValence",
        "MagpieData mean NdValence",
        "MagpieData mean AtomicWeight",
    ]]
    Y = df[[
        "zT"
    ]]
    Y.loc[Y["zT"] >= 0.08, "zT"] = 1
    Y.loc[Y["zT"] < 0.08, "zT"] = 0

    # return (
    #     x.to_numpy().astype('float') for x in train_test_split(X, Y, test_size=0.3)
    # )
    return train_test_split(X, Y, test_size=0.3)


def validate(predicted, YTest):
    acc = metrics.accuracy_score(YTest, predicted)
    pre = metrics.precision_score(YTest, predicted)
    recall = metrics.recall_score(YTest, predicted)
    f1 = metrics.f1_score(YTest, predicted)
    print("acc:", acc, " pre:", pre, " recall:", recall, " f1:", f1)
