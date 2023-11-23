# DATE: Nov 21, 2023
# Author: Kutay Eken

import numpy as np
import pandas as pd
import math
from numpy.linalg import norm
from scipy.optimize import minimize
from tqdm.auto import tqdm
import pickle

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

def primal_predict(x_data,w):
    y_pred = []
    for index, row in x_data.iterrows():
        row_list = row.to_numpy()
        y_pred.append(np.sign(np.dot(w.transpose(),row_list)))
    return y_pred

def equalityConstraint(alphas,y_data):
    return np.sum(np.multiply(alphas,y_data)) # = 0 EQUALITY CONSTRAINT

def dual_objective(alphas,xTx,yy):
    aa = np.outer(alphas, alphas)
    return 0.5 * (np.sum(yy * aa * xTx)) - np.sum(alphas)

def find_optimal(x_data,c,y_data,xTx,yy):
    init_guess = np.zeros(len(x_data.index))
    cons = {'type':'eq', 'fun': equalityConstraint, 'args': (y_data,)}
    bnds = [(0,c) for i in range(len(x_data.index))]
    result = minimize(dual_objective, init_guess, args=(xTx,yy),bounds=bnds, constraints=cons, method='SLSQP')

    return result.x

def find_learned_weights(optimal_alphas, y_data,x_data):
    w = np.zeros(len(x_data.columns))
    for index,row in x_data.iterrows():
        x_i = row.to_numpy()
        y_i = y_data[index]
        a_i = optimal_alphas[index]
        arr = x_i * a_i * y_i
        w = np.add(w,arr)

    return w

def find_bias(weights,x_data,y_data):
    b_arr = np.zeros(len(x_data.index))
    wT = weights.transpose()
    for index,row in x_data.iterrows():
        y_j = y_data[index]
        x_j = row.to_numpy()
        b_arr[index] = y_j * np.dot(wT,x_j)

    b = sum(b_arr) / len(b_arr)
    return b

def predict_dual(weights,bias,x_data):
    y_pred = []
    for index, row in x_data.iterrows():
        row_list = row.to_numpy()
        pred = np.sign(np.dot(weights.transpose(),row_list) + bias)
        y_pred.append(pred)
    
    return y_pred

def findSV(alphas,x_data,y_data):
    indices = np.where(alphas > 0)[0]
    x_vals = x_data.iloc[indices].to_numpy().tolist()
    y_vals = y_data[indices]
    sv_list = list(zip(x_vals,y_vals))
    sv_alphas = alphas[indices]
    # for i in range(len(alphas)):
    #     alpha = alphas[i]
    #     if alpha > 0: # Support Vector
    #         sv_alphas.append(alpha)
    #         x_val = x_data.iloc[i,:].tolist()
    #         y_val = y_data[i]
    #         sv_list.append((x_val,y_val))
    return sv_list, sv_alphas

def predict_kernel(alpha_sv,bias,support_vector,gamma,x_data):
    y_pred = []
    for index,row in x_data.iterrows():
        x_new = row.to_numpy()
        pred = 0
        for i in range(len(support_vector)):
            sv = support_vector[i]
            x_sv = sv[0]
            y_sv = sv[1]
            pred += alpha_sv[i] * y_sv * gaussian(x_new,x_sv,gamma) + bias
        y_pred.append(np.sign(pred))
    return y_pred

def gaussian(x_i,x_j,gamma):
    diff_norm = norm(x_i - x_j) ** 2
    val =  - diff_norm / gamma
    return math.exp(val)

def find_optimal_kernel(x_data,c,y_data,gamma):
    init_guess = np.zeros(len(x_data.index))
    cons = {'type':'eq', 'fun': equalityConstraint, 'args': (y_data,)}
    bnds = [(0,c) for i in range(len(x_data.index))]
    result = minimize(kernel_objective, init_guess, args=(x_data,y_data,gamma),bounds=bnds, constraints=cons, method='SLSQP')

    return result.x

def kernel_objective(alphas,x_data,y_data,gamma):
    x_data = x_data.to_numpy()
    # kernel_arr = np.zeros((len(x_data.index), len(x_data.index)))
    # for i,rowi in x_data.iterrows():
    #     x_i = rowi.to_numpy()
    #     for j,rowj in x_data.iterrows():
    #         x_j = rowj.to_numpy()
    #         kernel_arr[i][j] = gaussian(x_i,x_j,gamma)

    sum_val = np.sum((x_data[:, None] - x_data) ** 2, axis=-1)
    kernel_arr = np.exp(- sum_val / gamma)

    yy = np.outer(y_data,y_data)
    aa = np.outer(alphas, alphas)
    return 0.5 * (np.sum(yy * aa * kernel_arr)) - np.sum(alphas)

def runq2():
    train,test,y_train,y_test = set_data()
    print("-------------------- QUESTION 2A --------------------")
    c_list = [(100/873), (500/873), (700/873)]
    for c in c_list:
        print("C = ", c)
        w_a = primal(train,y_train,100,c,0)
        # PREDICT TEST DATA
        predictions_test_a = np.array(primal_predict(test,w_a))
        error_test_a = np.sum(y_test != predictions_test_a) / len(y_test)
        # PREDICT TRAINING DATA
        predictions_train_a = np.array(primal_predict(train,w_a))
        error_train_a = np.sum(y_train != predictions_train_a) / len(y_train)
        print("LEARNED WEIGHTS: ", w_a)
        print("TRAINING ERROR: ", error_train_a)
        print("TESTING ERROR: ", error_test_a)

    print("-------------------- QUESTION 2B --------------------")
    for c in c_list:
        print("C = ", c)
        w_b = primal(train,y_train,100,c,1)
        # PREDICT TEST DATA
        predictions_test_b = np.array(primal_predict(test,w_b))
        error_test_b = np.sum(y_test != predictions_test_b) / len(y_test)
        # PREDICT TRAINING DATA
        predictions_train_b = np.array(primal_predict(train,w_b))
        error_train_b = np.sum(y_train != predictions_train_b) / len(y_train)
        print("LEARNED WEIGHTS: ", w_b)
        print("TRAINING ERROR: ", error_train_b)
        print("TESTING ERROR: ", error_test_b)

    print("-------------------- QUESTION 2C --------------------")
    print("WEIGHT DIFFERENCE: ", w_a - w_b)
    print("TRAINING ERROR DIFFERENCE: ", error_train_a - error_train_b)
    print("TESTING ERROR DIFFERENCE: ", error_test_a - error_test_b)

def runq3a():
    print("-------------------- QUESTION 3A --------------------")
    train,test,y_train,y_test = set_data() 
    c_list = [(100/873), (500/873), (700/873)]  
    xTx =  np.zeros((872, 872))
    yy = np.outer(y_train,y_train)
    for i,rowi in (train.iterrows()):
        for j,rowj in train.iterrows():
            xTx[i][j] = (np.matmul(rowi.to_numpy().transpose(),rowj.to_numpy()))
  
    for c in tqdm(c_list):
        optimal_alpha = find_optimal(train,c,y_train,xTx,yy)
        learned_w = find_learned_weights(optimal_alpha,y_train,train) 
        learned_b = find_bias(learned_w,train,y_train)
        predictions = predict_dual(learned_w,learned_b,test)
        error_test = np.sum(y_test != predictions) / len(y_test)
        predictions_train = predict_dual(learned_w,learned_b,train)
        error_train = np.sum(y_train != predictions_train) / len(y_train)
        print("C: ", c)
        print("WEIGHTS: ", learned_w)
        print("BIAS: ", learned_b)
        print("TESTING ERROR: ", error_test)
        print("TRAINING ERROR: ", error_train)

def runq3b():
    print("-------------------- QUESTION 3B --------------------")
    train,test,y_train,y_test = set_data() 
    c_list = [(100/873), (500/873), (700/873)]  
    gamma_vals = [0.1,0.5,1,5,100]
    q3c_arr1 = []
    q3c_arr2 = []
    q3c_arr3 = []

    for gamma in gamma_vals:
        for c in tqdm(c_list, unit='value'):
            optimal_alpha = find_optimal_kernel(train,c,y_train,gamma)
            support_vectors,sv_alphas = findSV(optimal_alpha,train,y_train)
            if c == (100/873):
                q3c_arr1.append(support_vectors)
            elif c == (500/873):
                q3c_arr2.append(support_vectors)
            else:
                q3c_arr3.append(support_vectors)

            learned_w = find_learned_weights(optimal_alpha,y_train,train) 
            learned_b = find_bias(learned_w,train,y_train)
            print("W: ",  learned_w)
            predictions = predict_kernel(sv_alphas,learned_b,support_vectors,gamma,test)
            error_test = np.sum(y_test != predictions) / len(y_test)
            predictions_train = predict_kernel(sv_alphas,learned_b,support_vectors,gamma,train)
            error_train = np.sum(y_train != predictions_train) / len(y_train)
            print("C: ", c)
            print("GAMMA: ", gamma)
            print("TESTING ERROR: ", error_test)
            print("TRAINING ERROR: ", error_train)
    
def runq3c():
    print("-------------------- QUESTION 3C --------------------")
    train,test,y_train,y_test = set_data() 
    c_list = [(100/873), (500/873), (700/873)]  
    gamma_vals = [0.1,0.5,1,5,100]
    all_svs = [0] * 5

    for gamma in gamma_vals:
        print("GAMMA: ", gamma)
        i = gamma_vals.index(gamma)
        for c in tqdm(c_list, unit='value'):
            print("C: ", c)
            optimal_alpha = find_optimal_kernel(train,c,y_train,gamma)
            support_vectors,sv_alphas = findSV(optimal_alpha,train,y_train)
            if c == (500/873):
                all_svs[i] = support_vectors
            print("SV COUNT: ", len(support_vectors))
    
    sv01 = set(all_svs[0])
    sv05 = set(all_svs[1])
    sv1 = set(all_svs[2])
    sv5 = set(all_svs[3])
    sv100 = set(all_svs[4])

    print("0.1 - 0.5: ", len(sv01.intersection(sv05)))
    print("0.5 - 1: ",len(sv05.intersection(sv1)))
    print("1 - 5: ", len(sv1.intersection(sv5)))
    print("5 - 100: ", len(sv5.intersection(sv100)))

runq2()
runq3a()
runq3b()
runq3c()
