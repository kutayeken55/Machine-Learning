import pandas as pd
import numpy as np


# Three different versions of perceptron will be implemented.
# 1. Standard 2. Voted 3. Average
def set_data():
    train = pd.read_csv("train.csv", header=None)
    test = pd.read_csv("test.csv", header=None)
    train.columns = ["varaince","skewness","curtosis","entropy","label"]
    test.columns = ["varaince","skewness","curtosis","entropy","label"]
    y_train = train["label"].tolist()
    y_test = test["label"].tolist()
    train = train.drop("label",axis=1)
    test = test.drop("label",axis=1)

    y_train = np.array([-1 if val == 0 else val for val in y_train])
    y_test = np.array([-1 if val == 0 else val for val in y_test])

    return train,test,y_train,y_test

def standard(x_data,y_data,epoch):
    w = np.zeros(len(x_data.columns))
    for i in range(epoch):
        shuffled = x_data.sample(frac = 1)
        for index,row in x_data.iterrows():
            row_list = row.to_numpy()
            y_i = y_data[index]
            val = y_i * np.dot(w.transpose(),row_list)
            if val <= 0:
                w = w + y_i * row_list        
    return w

def predict(x_data,w):
    y_pred = []
    for index, row in x_data.iterrows():
        row_list = row.to_numpy()
        y_pred.append(np.sign(np.dot(w.transpose(),row_list)))
    return y_pred


train,test,y_train,y_test = set_data()
w_std = (standard(test,y_test,10))
pred_std = np.array(predict(test,w_std))
similarity_std = np.sum(y_test == pred_std) / len(y_test)
print("----------STANDARD PERCEPTRON RESULTS:----------")
print("LEARNED WEIGHT VECTOR: ", w_std)
print("AVERAGE PREDICTION ERROR ON TEST DATASET: ",1 - similarity_std)