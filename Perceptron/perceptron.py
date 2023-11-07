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

    # train["label"] = y_train
    return train,test,y_train,y_test

def standard(x_data,y_data,epoch):
    w = np.zeros(len(x_data.columns))
    for i in range(epoch):
        x_data = x_data.sample(frac = 1)
        for index,row in x_data.iterrows():
            row_list = row.to_numpy()
            y_i = y_data[index]
            val = y_i * np.dot(w.transpose(),row_list)
            if val <= 0:
                w = w + y_i * row_list        
    return w

def averaged(x_data,y_data,epoch):
    w = np.zeros(len(x_data.columns))
    a = np.zeros(len(x_data.columns))
    for i in range(epoch):
        for index, row in x_data.iterrows():
            row_list = row.to_numpy()
            y_i = y_data[index]
            val = y_i * np.dot(w.transpose(),row_list)
            if val <= 0:
                w = w + y_i * row_list 
            a = a + w 
    return a,w 

def voted(x_data,y_data,epoch):
    w_init = np.zeros(len(x_data.columns))
    m = np.zeros(len(x_data.columns))
    result = []
    w_current = w_init
    for i in range(epoch):
        for index, row in x_data.iterrows():
            row_list = row.to_numpy()
            y_i = y_data[index]
            val = y_i * np.dot(w_current.transpose(),row_list)
            if val <= 0:
                w_next = w_current + y_i * row_list
                m = m + 1
                c_current = 1
                w_current = w_next
            else:
                c_current = c_current + 1
            result.append((w_current.tolist(),c_current))

    return result           

def predict(x_data,w):
    y_pred = []
    for index, row in x_data.iterrows():
        row_list = row.to_numpy()
        y_pred.append(np.sign(np.dot(w.transpose(),row_list)))
    return y_pred

def predict_vote(x_data,w_c):
    k = len(w_c)
    w_list = list(zip(*w_c))[0]
    c_list = list(zip(*w_c))[1]
    y_pred = []
    for index, row in x_data.iterrows():
        res = 0
        row_list = row.to_numpy()
        for i in range(k):
            # CONTINUE
            c_i = c_list[i]
            w_i = w_list[i]
            w_iT  = np.array(w_i).transpose()
            res += c_i * np.sign(np.dot(w_iT,row_list))
        y_pred.append(np.sign(res))

    return y_pred


train,test,y_train,y_test = set_data()

# w_std = (standard(train,y_train,10))
# pred_std = np.array(predict(test,w_std))
# similarity_std = np.sum(y_test == pred_std) / len(y_test)
# print("----------STANDARD PERCEPTRON RESULTS:----------")
# print("LEARNED WEIGHT VECTOR: ", w_std)
# print("AVERAGE PREDICTION ERROR ON TEST DATASET: ",1 - similarity_std)
# print("ERROR PERCENTAGE: ",(1 - similarity_std) * 100)
# print("------------------------------------------------")


# a_ave,w_ave = averaged(train,y_train,10)
# pred_ave = np.array(predict(test,a_ave))
# similarity_ave = np.sum(y_test == pred_ave) / len(y_test)
# print("----------AVERAGED PERCEPTRON RESULTS:----------")
# print("LEARNED WEIGHT VECTOR: ", w_ave)
# print("AVERAGE PREDICTION ERROR ON TEST DATASET: ",1 - similarity_ave)
# print("ERROR PERCENTAGE: ",(1 - similarity_ave) * 100)
# print("------------------------------------------------")

w_c_vote = voted(train,y_train,10)
seen_first_items = set()
unique_list = []

for item in w_c_vote:
    first_item = tuple(item[0])
    
    # Check if the first item is already in the set
    if first_item not in seen_first_items:
        seen_first_items.add(first_item)
        unique_list.append(item)




pred_vote = np.array(predict_vote(test,w_c_vote))
similarity_vote = np.sum(y_test == pred_vote) / len(y_test)
print("----------VOTED PERCEPTRON RESULTS:----------")
print("WEIGHT VECTORS AND COUNTS: ", (unique_list))
print("AVERAGE PREDICTION ERROR ON TEST DATASET: ",1 - similarity_vote)
print("ERROR PERCENTAGE: ",(1 - similarity_vote) * 100)
print("---------------------------------------------")
print(len(unique_list))