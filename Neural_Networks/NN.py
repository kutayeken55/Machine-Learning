import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

    
class NeuralNetwork(nn.Module):
    def __init__(self, activation_function, width, depth,func_name):
        super(NeuralNetwork,self).__init__()
        in_dim = 4
        layers_list = []

        # input layer
        layers_list.append(nn.Linear(in_dim,width))
        layers_list.append(activation_function)

        for i in range(1,depth):
            if i <= depth - 1: # hidden layer
                layers_list.append(nn.Linear(width,width))
                layers_list.append(activation_function)

        layers_list.append(nn.Linear(width,1))

        self.net = nn.ModuleList(layers_list)

        if func_name == 'tanh':
            self.init_tanh_weights()
        else:
            self.init_RELU_weights()

    def init_tanh_weights(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def init_RELU_weights(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias,0)

    def forward(self, X):
        output = X
        for layer in self.net:
            output = layer(output)
        
        return output

def set_data():
    train = pd.read_csv("train.csv", header=None)
    test = pd.read_csv("test.csv", header=None)
    train.columns = ["varaince","skewness","curtosis","entropy","label"]
    test.columns = ["varaince","skewness","curtosis","entropy","label"]
    y_train = train["label"].tolist()
    y_test = test["label"].tolist()
    train = train.drop("label",axis=1)
    test = test.drop("label",axis=1)

    # y_train = np.array([-1 if val == 0 else val for val in y_train])
    # y_test = np.array([-1 if val == 0 else val for val in y_test])

    return train.values,test.values,y_train,y_test

def run2e():
    train,test,y_train,y_test = set_data() 
    train = torch.FloatTensor(train)
    test = torch.FloatTensor(test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    criterion = nn.BCEWithLogitsLoss()
    num_epoch = 100

    activation_fun = ["relu", "tanh"]
    depths = [3,5,9]
    widths = [5,10,25,50,100]

    model = NeuralNetwork(nn.Tanh(),3,3,'tanh')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epoch):
        for item in train:
            optimizer.zero_grad()
            y_pred = model(item).flatten()
            loss_val = criterion(y_pred,y_train)
            loss_val.backward()
            optimizer.step()

    with torch.no_grad():
        predictions = model(train)
        print(predictions)
        error = np.sum(y_train.tolist() != predictions.tolist()) / len(y_train.numpy())
        print("ERROR: ", error)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def createq2NN(hidden_unit):
    # Create the neural network for 2a as a dictionary
    # keys: tuple (node, layer)
    # values: list of tuples (edge, destination)
    y_est = None

    # layer 0 of network is input variables
    node_layer_key = [('x0',0), ('x1',0), ('x2',0)]
    edge_dest_val = []
    
    # creating variables based on hidden_units
    for layer in range(1,3): # for each hidden layers in the network
        for unit in range(hidden_unit): # creating different variables for hidden layer nodes
            var_name = f"z_{unit}_{layer}" #creating new variable                
            node_layer_key.append((var_name,layer))


    for (node_var,node_layer) in node_layer_key:
        node_edge = []
        if node_var[0] == 'x':
            node_sub = int(node_var[-1])
        else:
            node_sub = int(node_var[-3])
        if node_layer < 2:
            edge_num = hidden_unit - 1
        else:
            edge_num = 1
        
        for edge in range(edge_num):
            edgeVar = f"w_{node_sub}{edge+1}_{node_layer+1}"
            # edge_value = var_val_dict[edgeVar]
            dest_name = f"z_{edge + 1}_{node_layer + 1}"
            if edge_num == 1:
                edgeVar = f"w_{node_var[-3]}{1}_{node_layer+1}"
                dest_name = "y"
            
            node_edge.append((edgeVar,dest_name))
        edge_dest_val.append(node_edge)

    # adding output variable to the net
    # node_layer_key.append((y_est,3))
    # edge_dest_val.append(None)

    # for i in range(3):
    #     node_name = node_layer_key[i][0]
    #     node_layer = node_layer_key[i][1]
    #     node_layer_key[i] = (var_val_dict[node_name],node_layer)
    # print(node_layer_key[-5])
    # print()
    # print(edge_dest_val[-5])
    return node_layer_key, edge_dest_val

def forward_pass(unknown_nodes, neural_net,edge_val,var_val):
    # edge_dest_list = list(neural_net.values())
    var_layer_list = list(neural_net.keys())
    # computed_vals = []
    unknown_layers = []
    for n in unknown_nodes:
        unknown_layers.append(int(n[-1]))
    
    unknown_layers = sorted(list(set(unknown_layers)))

    for unknown_layer in unknown_layers:
        for unknown in unknown_nodes:
            if unknown[-1] == str(unknown_layer):
                sigmoid_val = 0
                incoming_nodes = []
                incoming_edges = []
                for node in var_layer_list:
                    for (edge,dest) in neural_net[node]:
                        if dest == unknown:
                            incoming_nodes.append(node)
                            incoming_edges.append(edge)
                for i in range(len(incoming_nodes)):
                    incoming_node_val = var_val[incoming_nodes[i][0]]
                    incoming_edge_val = edge_val[incoming_edges[i]]
                    sigmoid_val += (incoming_edge_val * incoming_node_val)
                computed = sigmoid(sigmoid_val)
                var_val[unknown] = computed

    y_incoming_edges = []
    y_incoming_nodes = []
    y_sigmoid_val = 0
    for node in var_layer_list:
        for (edge,dest) in neural_net[node]:
            if dest == 'y':
                y_incoming_edges.append(edge)
                y_incoming_nodes.append(node)
    for i in range(len(y_incoming_nodes)):
        incoming_node_val = var_val[y_incoming_nodes[i][0]]
        incoming_edge_val = edge_val[y_incoming_edges[i]]
        y_sigmoid_val += (incoming_edge_val * incoming_node_val)

    var_val['y'] = (y_sigmoid_val)

    return var_val

def isOutPutLayer(weight):
    return weight[-1] == str(3)

def findStartingNodeGivenEdge(edge,endPoint,neural_net):
    for node in list(neural_net.keys()):
        if (edge,endPoint) in neural_net[node]:
            return node[0]

def findDependents(neural_net, given_edge):
    result = []
    for node in list(neural_net.keys()):
        for (edge,endPoint) in neural_net[node]:
            if edge == given_edge:
                result.append(endPoint)

    return list(set(result))

def computeOutPutLayer(edge,neural_net,delta,var_val,edge_val):
    arrived_from = findStartingNodeGivenEdge(edge,'y',neural_net)
    node_val = var_val[arrived_from]
    return delta * node_val

def back_propagation(x0,x1,x2,y,y_star,var_val, edge_val, neural_net):
    # square loss function
    # L_func = 0.5 * (y - y_star)**2

    # delta L_func over delta y
    # delta_Ly = (y - y_star)
    # derivatives = []

    # for edge in list(edge_val.keys()):
    #     if isOutPutLayer(edge):
    #         # print(edge)
    #         # print()
    #         # print(computeOutPutLayer(edge,neural_net,delta_Ly,var_val,edge_val))
    #         derivatives.append(computeOutPutLayer(edge,neural_net,delta_Ly,var_val,edge_val))
    #     else:
    #         dependent_nodes = findDependents(neural_net,edge)
    #         derivatives.append(computeHiddenLayer(edge,neural_net,delta_Ly,var_val,edge_val,dependent_nodes))

    ## FOLLOWING SLIDES BELOW:
    for edge in list(edge_val.keys()):
        h = int(edge[-1])
        n = int(edge[-3])
        m = int(edge[-4])
        starting_node = (f"z_{n}_{h}",h)
        ending_node = "y"
        path_list = find_all_paths("z_0_2",ending_node,neural_net,["z_0_2"],[])
        print(path_list)

def find_all_paths(starting_node,ending_node,neural_net,current_path,all_paths):
    if (starting_node == ending_node):
        current_path.append(ending_node)
        all_paths.append(current_path)
        return current_path
    else:
        if starting_node[0] == 'x':
            layer = 0
        else:
            layer = int(starting_node[-1])
        for (edge,dest) in neural_net[(starting_node,layer)]:
            current_path.append(find_all_paths(dest,ending_node,neural_net,current_path,all_paths))

# def run2a():
# vars = ["x0","x1","x2","z_0_1","z_0_2"]
# vals = [1,1,1,1,1]
# var_val_dict = dict(zip(vars, vals))

# edges = ["w_01_1","w_02_1","w_11_1","w_12_1","w_21_1","w_22_1","w_01_2","w_02_2","w_11_2","w_12_2","w_21_2","w_22_2","w_01_3","w_11_3","w_21_3"]
# eVals = [-1,1,-2,2,-3,3,-1,1,-2,2,-3,3,-1,2,-1.5]
# edge_val_dict = dict(zip(edges,eVals))

# key,val = createq2NN(3)
# my_nn = dict(zip(key,val))
# nodes = [tuples[0] for tuples in key]
# hidden_nodes = list(set(vars) ^ set(nodes))

# var_val_dict = forward_pass(hidden_nodes, my_nn,edge_val_dict,var_val_dict)
# back_propagation(1,1,1,var_val_dict['y'],1,var_val_dict,edge_val_dict,my_nn)
# print(key[0])
# print()
# print(val[0])x0,x1,x2,y,y_star,var_val, edge_val, neural_net

run2e()
