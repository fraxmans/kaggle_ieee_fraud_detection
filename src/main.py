import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
import lightgbm as lgb
import matplotlib.pyplot as plt

import copy
import datetime
from fraud_preprocessing import reduce_mem_usage, transaction_category_col, transaction_float_col, transaction_usecols, identity_category_col, label_encoding

params = {
            "task": "train",
            "metric": "auc",
            "objective": "binary",
            "num_iterations": 1000,
            #"early_stopping_round": 10,
            "learning_rate": 0.01,
            "num_leaves": 60,
            "min_data_in_leaf": 60,
            "max_depth": -1,
            "bagging_freq": 1,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "max_bin": 63,
            "lambda_l1": 0.1,
            "is_unbalance": "true"
        }

def print_feature_importance(feature_name, feature_importance):

    result = [[name, importance] for name, importance in zip(feature_name, feature_importance)]
    result = sorted(result, key=lambda x: x[1])

    for e in result:
        print(e)

def feature_engineering(train_transaction, test_transaction, category_col):
    drop_col = ["id_07", "id_08", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27", "id_18", "D7", "dist2", "D13", "D14"]

    for col in train_transaction.columns:
        if(1.0 * train_transaction[col].isnull().sum() / train_transaction.shape[0] > 0.5):
            if(not col in drop_col):
                drop_col.append(col)

    train_transaction.drop(drop_col, inplace=True, axis=1)
    test_transaction.drop(drop_col, inplace=True, axis=1)
    for col in drop_col:
        if(col in category_col):
            category_col.remove(col)

    dataset = train_transaction.append(test_transaction, ignore_index=True, sort=False)
    dataset = label_encoding(dataset, category_col)

    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    dataset["Date"] = dataset["TransactionDT"].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    dataset["Weekdays"] = dataset["Date"].dt.dayofweek
    dataset["Hours"] = dataset["Date"].dt.hour
    dataset["Days"] = dataset["Date"].dt.day
    dataset.drop("Date", inplace=True, axis=1)

    dummy_col = ["card4", "card6", "M1", "M2", "M3", "M4", "M6", "Weekdays", "Hours", "Days"]
    dataset = pd.get_dummies(dataset, columns=dummy_col)
    for col in dummy_col:
        if(col in category_col):
            category_col.remove(col)

    train_transaction = dataset[:train_transaction.shape[0]]
    test_transaction = dataset[train_transaction.shape[0]:]

    return train_transaction, test_transaction

def train(train_transaction, label, test_transaction, category_col):
    train_transaction.drop("TransactionID", inplace=True, axis=1)
    ID = test_transaction["TransactionID"].copy()
    test_transaction.drop("TransactionID", inplace=True, axis=1)

    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits)
    clf = None
    avg_roc = 0
    importance = 0
    test_predict = 0

    for train_idx, valid_idx in kf.split(train_transaction, label):
        x_train = train_transaction.iloc[train_idx]
        y_train = label.iloc[train_idx]
        x_valid = train_transaction.iloc[valid_idx]
        y_valid = label.iloc[valid_idx]

        clf = lgb.LGBMClassifier(num_leaves=60, min_child_samples=60, subsample_freq=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.8, metric=None, learning_rate=0.1, n_estimators=100)
        clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], categorical_feature=category_col, eval_metric="auc")

        valid_predict = clf.predict_proba(x_valid)[:, 1]
        roc = roc_auc_score(y_valid, valid_predict)

        avg_roc += roc / n_splits
        importance += clf.feature_importances_ / n_splits
        test_predict += clf.predict_proba(test_transaction)[:, 1] / n_splits

    print_feature_importance(clf.booster_.feature_name(), importance)
    print("Average roc socre: %f" % avg_roc)
    result = pd.DataFrame({"TransactionID": ID, "isFraud": test_predict})
    result.to_csv("submission.csv", index=False)

def cv(train_transaction, label, category_col):
    train_transaction.drop("TransactionID", inplace=True, axis=1)
    trainset = lgb.Dataset(train_transaction, label=label, categorical_feature=category_col)
    eval_hist = lgb.cv(params, trainset, verbose_eval=True)
    print("Best score: %f" % max(eval_hist["auc-mean"]))

def load_data(fname, isTest=False, usecols=None):
    cols = copy.copy(usecols)
    if(isTest and not (usecols == None)):
        cols.remove("isFraud")

    data = pd.read_csv(fname, usecols=cols)

    if(isTest and (usecols == None)):
        rename_dict = {}
        for i in range(1, 39):
            original = "id-%02d" % i
            new = "id_%02d" % i
            rename_dict[original] = new
        data.rename(columns=rename_dict, inplace=True)

    return data

def main():
    train_transaction = load_data("data/train_transaction.csv", usecols=transaction_usecols)
    train_identity = load_data("data/train_identity.csv")
    test_transaction = load_data("data/test_transaction.csv", isTest=True, usecols=transaction_usecols)
    test_identity = load_data("data/test_identity.csv", isTest=True)

    train_transaction = reduce_mem_usage(train_transaction)
    train_identity = reduce_mem_usage(train_identity)
    test_transaction = reduce_mem_usage(test_transaction)
    test_identity = reduce_mem_usage(test_identity)

    label = train_transaction["isFraud"].copy()
    train_transaction.drop("isFraud", axis=1, inplace=True)
    
    train_transaction = train_transaction.merge(right=train_identity, how="left", on="TransactionID")
    test_transaction = test_transaction.merge(right=test_identity, how="left", on="TransactionID")
    tmp = [transaction_category_col, identity_category_col]
    category_col = [item for sublist in tmp for item in sublist]
    
    train_transaction, test_transaction = feature_engineering(train_transaction ,test_transaction, category_col)

    #cv(train_transaction, label, category_col)
    train(train_transaction, label, test_transaction, category_col)

main()
