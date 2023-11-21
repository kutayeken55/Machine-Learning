# DATE: Nov 21, 2023
# Author: Kutay Eken

import numpy as np
import pandas as pd

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

def primal(x_data,y_data,epoch,c,schedule):
    w =  np.zeros(len(x_data.columns)) # initialize weights
    gamma = 0.01
    a = 0.5
    N = len(x_data.index)
    for t in range(epoch):
        if schedule == 0:
            gamma_t = gamma / ((1 + ((gamma * t) / a)))
        else:
            gamma_t = gamma / (1 + t)
        x_data = x_data.sample(frac = 1) # shuffle the data
        for index,row in x_data.iterrows():
            x_i = row.to_numpy()
            y_i = y_data[index]
            val = y_i * np.dot(w.transpose(),x_i)
            if val <= 1:
                w = w - gamma_t * np.array([w[0],0,0,0]) + gamma_t * c * N * y_i * x_i
            else:
                w[0] = (1 - gamma_t) * w[0]
    
    return w

def predict(x_data,w):
    y_pred = []
    for index, row in x_data.iterrows():
        row_list = row.to_numpy()
        y_pred.append(np.sign(np.dot(w.transpose(),row_list)))
    return y_pred

def runq2():
    train,test,y_train,y_test = set_data()
    print("-------------------- QUESTION 2A --------------------")
    c_list = [(100/873), (500/873), (700/873)]
    for c in c_list:
        print("C = ", c)
        w_a = primal(train,y_train,100,c,0)
        # PREDICT TEST DATA
        predictions_test_a = np.array(predict(test,w_a))
        error_test_a = np.sum(y_test != predictions_test_a) / len(y_test)
        # PREDICT TRAINING DATA
        predictions_train_a = np.array(predict(train,w_a))
        error_train_a = np.sum(y_train != predictions_train_a) / len(y_train)
        print("TRAINING ERROR: ", error_train_a)
        print("TESTING ERROR: ", error_test_a)

    print("-------------------- QUESTION 2B --------------------")
    for c in c_list:
        print("C = ", c)
        w_b = primal(train,y_train,100,c,1)
        # PREDICT TEST DATA
        predictions_test_b = np.array(predict(test,w_b))
        error_test_b = np.sum(y_test != predictions_test_b) / len(y_test)
        # PREDICT TRAINING DATA
        predictions_train_b = np.array(predict(train,w_b))
        error_train_b = np.sum(y_train != predictions_train_b) / len(y_train)
        print("TRAINING ERROR: ", error_train_b)
        print("TESTING ERROR: ", error_test_b)

    print("-------------------- QUESTION 2C --------------------")
    print("WEIGHT DIFFERENCE: ", w_a - w_b)
    print("TRAINING ERROR DIFFERENCE: ", error_train_a - error_train_b)
    print("TESTING ERROR DIFFERENCE: ", error_test_a - error_test_b)

runq2()