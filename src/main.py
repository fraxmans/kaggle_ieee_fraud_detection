import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
import lightgbm as lgb
import matplotlib.pyplot as plt

import copy
from common import reduce_mem_usage, transaction_category_col, transaction_float_col, transaction_usecols, identity_category_col, label_encoding

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
    train_transaction = dataset[:train_transaction.shape[0]]
    test_transaction = dataset[train_transaction.shape[0]:]

    return train_transaction, test_transaction

def test(bst, test_transaction, category_col):
    ID = test_transaction["TransactionID"].copy()
    test_transaction.drop("TransactionID", inplace=True, axis=1)
    result = bst.predict(test_transaction)

    submission = pd.DataFrame({"TransactionID": ID, "isFraud": result})
    submission.to_csv("submission.csv", index=False)

def train(train_transaction, label, category_col):
    train_transaction.drop("TransactionID", inplace=True, axis=1)
    trainset = lgb.Dataset(train_transaction, label=label, categorical_feature=category_col)
    bst = lgb.train(params, trainset)
    return bst

def cv(train_transaction, label, category_col):
    train_transaction.drop("TransactionID", inplace=True, axis=1)
    trainset = lgb.Dataset(train_transaction, label=label, categorical_feature=category_col)
    eval_hist = lgb.cv(params, trainset, verbose_eval=True)
    print(eval_hist)

def load_data(fname, isTest=False, usecols=None):
    cols = copy.copy(usecols)
    if(isTest and not (usecols == None)):
        cols.remove("isFraud")
    
    data = pd.read_csv(fname, usecols=cols)

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

    cv(train_transaction, label, category_col)
    #bst = train(train_transaction, label, category_col)
    #test(bst, test_transaction, category_col)

main()
