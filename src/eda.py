import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fraud_preprocessing import reduce_mem_usage, transaction_category_col, transaction_float_col, c_col, transaction_usecols, identity_category_col

import copy

def fraud_ratio_in_id_33(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["id_33"])["isFraud"].mean()
    print(res)
    ax = sns.barplot(x=res.index, y=res)
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment="left")
    plt.show()

def fraud_ratio_in_id_31(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["id_31"])["isFraud"].mean()
    print(res)
    ax = sns.barplot(x=res.index, y=res)
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment="left")
    plt.show()

def fraud_ratio_in_id_30(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["id_30"])["isFraud"].mean()
    print(res)
    ax = sns.barplot(x=res.index, y=res)
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment="left")
    plt.show()

def fraud_ratio_in_id_23(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["id_23"])["isFraud"].mean()
    print(res)
    ax = sns.barplot(x=res.index, y=res)
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment="left")
    plt.show()

def transaction_float_feature_evaluation(df):
    tmp = copy.copy(transaction_float_col)
    tmp.append("isFraud")
    corr = df[tmp].corr()
    print(corr["isFraud"].pow(2).sort_values())
    print(corr[transaction_float_col].pow(2).sum(axis=1).sort_values())

def fraud_ratio_in_R_emaildomain(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["R_emaildomain"])["isFraud"].mean()
    print(res)
    ax = sns.barplot(x=res.index, y=res)
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment="left")
    plt.show()

def fraud_ratio_in_P_emaildomain(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["P_emaildomain"])["isFraud"].mean()
    print(res)
    ax = sns.barplot(x=res.index, y=res)
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment="left")
    plt.show()

def fraud_ratio_in_card6(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["card6"])["isFraud"].mean()
    print(res)
    sns.barplot(x=res.index, y=res)
    plt.show()

def fraud_ratio_in_card4(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["card4"])["isFraud"].mean()
    print(res)
    sns.barplot(x=res.index, y=res)
    plt.show()

def fraud_ratio_in_ProductCD(df):
    plt.figure(figsize=(20, 20))
    res = df.groupby(["ProductCD"])["isFraud"].mean()
    sns.barplot(x=res.index, y=res)
    plt.show()

def check_null(df):
    length = df.shape[0]
    result = []

    for col in df.columns:
        null_ratio = 100.0 * df[col].isnull().sum() / length
        result.append([col, null_ratio])

    result = sorted(result, key=lambda x: x[1])
    
    for col, ratio in result:
        print(col, ratio)

def load_data(fname, isTest=False, usecols=None):
    data = pd.read_csv(fname, usecols=usecols)

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
    train_transaction = reduce_mem_usage(train_transaction)

    train_identity = load_data("data/train_identity.csv")
    train_identity["id_30"].replace(regex={"Android.*": "Android", "chrome.*": "chrome", "iOS.*": "iOS", "Linux.*": "Linux", "Mac OS.*": "Mac OS", "Windows.*": "Windows"}, inplace=True)
    train_identity = reduce_mem_usage(train_identity)

    train_transaction = train_transaction.merge(right=train_identity, how="left")

main()

