import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random   
from numpy import linalg as LA

def calcCost(weights,data,y):
  res = 0
  for i in range(len(data.index)):
    sample = data.iloc[i].values.tolist()[:7]
    weights_np = np.array(weights)
    res += (y[i] - np.dot(weights_np.transpose(),sample)) ** 2
  return res * 0.5

def sgd(data,y):
  # initialize w,
  r = 0.002
  w_next = [-1] * 7
  w_current = [0] * 7
  cost = []

  for t in range(5000):
      i = random.randint(0,len(data.index) - 1)
      sample = data.iloc[i].values.flatten().tolist()[:7]
      x1 = sample[0]
      x2 = sample[1]
      x3 = sample[2]
      x4 = sample[3]
      x5 = sample[4]
      x6 = sample[5]
      x7 = sample[6]
      for w_i in w_current:
        j = w_current.index(w_i)
        w_np =  np.array(w_current)
        w_next[j] = w_i + r * (y[i] - np.dot(w_np.transpose(),[x1,x2,x3,x4,x5,x6,x7]))  * data.loc[i][j]
      w_current = w_next

      cost.append(calcCost(w_next,data,y))

  return cost,w_current

def bgd(data,y,r):
  # initialize w,
  w_next = [-1] * 7
  w_current = [0] * 7
  cost = []
  converge = False

  while not converge:
    for i in range(len(data.index)):
        sample = data.iloc[i].values.flatten().tolist()[:7]
        x1 = sample[0]
        x2 = sample[1]
        x3 = sample[2]
        x4 = sample[3]
        x5 = sample[4]
        x6 = sample[5]
        x7 = sample[6]
        for w_i in w_current:
          j = w_current.index(w_i)
          w_np =  np.array(w_current)
          w_next[j] = w_i + r * (y[i] - np.dot(w_np.transpose(),[x1,x2,x3,x4,x5,x6,x7]))  * data.loc[i][j]
        arr_diff = np.array(w_next) - np.array(w_current)
        normVal = LA.norm(arr_diff)
        w_current = w_next
        if normVal < 10 ** (-16):
          converge = True
        cost.append(calcCost(w_next,data,y))

  return cost,w_current


def runsgd():
    print("STOCHASTIC GRADIENT...")
    test = pd.read_csv("test.csv")
    test.columns = ["x1","x2","x3","x4","x5","x6","x7","y"]
    y_test = test["y"]
    train = pd.read_csv("train.csv")
    train.columns =["x1","x2","x3","x4","x5","x6","x7","y"]
    y_train = train["y"]
    cost,weights = sgd(train,y_train)
    print("LEARNED WEIGHTS: ", weights)
    plt.plot(cost,np.arange(5000))
    plt.xlabel("Cost")
    plt.ylabel("Iteration")
    plt.show()
    plt.savefig("Stochastic.pdf", format="pdf",bbox_inches = "tight")

    print("TEST COST USING LEARNED WEIGHTS: ",calcCost(weights,test,y_test))

def runbgd():
    print("BATCH GRADIENT...")
    test = pd.read_csv("test.csv")
    test.columns = ["x1","x2","x3","x4","x5","x6","x7","y"]
    y_test = test["y"]
    train = pd.read_csv("train.csv")
    train.columns =["x1","x2","x3","x4","x5","x6","x7","y"]
    y_train = train["y"]
    costs,weights = bgd(train,y_train,0.0015)
    print("LEARNED WEIGHTS: ", weights)
    plt.plot(costs,np.arange(len(costs)))
    plt.xlabel("Cost")
    plt.ylabel("Iteration")
    plt.show()
    plt.savefig("Batch.pdf", format="pdf",bbox_inches = "tight")
    print("TEST COST USING LEARNED WEIGHTS: ",calcCost(weights,test,y_test))

runsgd()
runbgd()