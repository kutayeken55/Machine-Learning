# Importing functions from MLTEST.py to call Decision Tree Algorithms
# import sys
# sys.path.append("DecisionTree")
# import MLTEST.py
import pandas as pd  
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

def find_info_gain(data_set, attribute, label, outcome_list):
  attribute_vals = data_set[attribute].unique()
  current_info = 0.0
  total_data_count = len(data_set.index)
  total_entropy = find_data_entropy_weights(data_set,label,outcome_list)

  for val in attribute_vals:
    df_sorted = data_set[data_set[attribute] == val]
    sorted_data_count = len(df_sorted.index)
    ent = find_data_entropy_weights(df_sorted,label,outcome_list)
    attribute_proportion = sorted_data_count / total_data_count
   # print(attribute, " entropy is " ,  ent)
    if pd.isna(ent) == False:
      current_info += attribute_proportion * ent
  return total_entropy - current_info

def find_data_entropy_weights(data_set, label, outcome_list):
    # data_set -> data set that the entropy will be calculated for.
    # label -> attribute that the outcome is displayed under 'y' for HW1 Q1.
    # outcome_list -> possible outcomes that can appear under the label. [0,1] for HW1 Q1.
    result = 0
    current_entropy = 0
    total_data_count = len(data_set.index) # finds the number of data in dataset to use it while calculating proprotion
    weights = data_set["Weights"]
    for outcome in outcome_list:
        proportion = sum(data_set.loc[data_set[label] == outcome, "Weights"])
        # print("Proportion for outcome ", outcome, " is: ", proportion)

        # proportion = len(data_set[data_set[label]==outcome]) / total_data_count
        # finding the proportion for each possible outcome
        current_entropy = -proportion * np.log2(proportion)

        if proportion != 0:
            # calculate the entropy for the current outcome
            result += current_entropy
        # adds to the entropy for the whole data set
    return result

def find_split_attribute(data_set, label, outcome_list):
    attribute_list = data_set.columns.drop(label)
    info_gains = []

    for attribute in attribute_list:
        info_gain_val = find_info_gain(data_set,attribute,label,outcome_list)
        info_gains.append(info_gain_val)

    max_gain = max(info_gains)
    split_attribute = attribute_list[info_gains.index(max_gain)]

    return split_attribute

def Decision_Stump(data, attributes, label, outcome_list, attribute_vals, tree, current_depth):
  if tree is None:
      tree = {}

  if current_depth == 2:
    leaf_node = max(set(data[label]), key=list(data[label]).count) 
    return leaf_node

  if len(set(data[label])) == 1: # all examples have same label
    if len(attributes) == 0: 
      leaf = max(set(data[label]), key=data[label].count) 
     # leaf node with most common value
    else:
        leaf = data[label].unique()[0]
      # leaf node with the only value
    return leaf
  else:
    root_node = {} # create a root node for the tree
    A = find_split_attribute(data, label, outcome_list)
    root_node[A] = {}
    for v in attribute_vals[A]:
      root_node[A][v] = {} # create a new tree branch for A=v
      S_v = data[data[A] == v] # S_v subset of examples where A = v
      if len(S_v) == 0: # if S_v is empty
        leaf_node = max(set(data[label]), key=list(data[label]).count) # leaf node with most common value of Label in S
        root_node[A][v] = leaf_node
      else:      
          root_node[A][v] = Id3_IG(S_v, attributes, label, outcome_list, attribute_vals,tree, current_depth + 1)
    return root_node

# # We have 'm' examples
# # D_t is a set of weights over the examples [D_t(1),...., D_t(m)]
# # initially uniform dist. weight is 1/m
# # Compute Info Gain to select the best feature

# # New weight  = sample weight * e^{amount of say} while increasing sample weight for incorrectly classified examples
# # New weight = sample weight * e^{- amount of say} while decreasing sample weight for correctly classified examples.
# # Amount of say  = 0.5 * log((1- Total Error) / Total Error)
# # NORMALIZE VALUES THEN
def adaBoost(t_iter,m):
    D_t = [1/m] * m # Initializing weights
    # for t in range(t_iter):
        # find a classifier h_t whose weighted classification error is better
        # compute its vote (amount of say):
        # alpha_t = 0.5 ln ((1 - e_t) / e_t)
        # Update the values of the weights for the training examples.
    
    # Return the final hypothesis,
    return 0
        

# initialize list of lists 
data = [['S','H','H','W','-',2/8],
        ['S','H','H','S','-',1/8],
        ['O','H','H','W','+',1/8],
        ['R','M','H','W','+',3/8],
        ['R','C','N','W','+',1/8]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns=['Outlook','Temperature','Humidity','Wind','Play?','Weights']) 
# find_data_entropy_weights(df,'Play?', ['+','-'])
print(find_split_attribute(df,'Play?', ['+','-']))
