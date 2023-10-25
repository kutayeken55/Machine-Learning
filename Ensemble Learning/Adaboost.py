import pandas as pd  
import numpy as np
import math
import warnings 
import statistics
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def find_info_gain(data_set, attribute, label, outcome_list):
  # print(data_set)
  attribute_vals = data_set[attribute].unique()
  current_info = 0.0
  total_data_count = sum(data_set["Weights"])
  total_entropy = find_data_entropy_weights(data_set,label,outcome_list)
  for val in attribute_vals:
    df_sorted = data_set[data_set[attribute] == val]
    sorted_data_count = sum(df_sorted["Weights"]) #len(df_sorted.index)
    ent = find_data_entropy_weights(df_sorted,label,outcome_list)
    attribute_proportion = sorted_data_count / total_data_count
    if pd.isna(ent) == False:
      current_info += attribute_proportion * ent

  # print("INFO GAIN FOR ATTRIBUTE: ", attribute, " IS ", total_entropy-current_info)
  return total_entropy - current_info

def find_data_entropy_weights(data_set, label, outcome_list):
  # data_set -> data set that the entropy will be calculated for.
  # label -> attribute that the outcome is displayed under 'y' for HW1 Q1.
  # outcome_list -> possible outcomes that can appear under the label. [0,1] for HW1 Q1.
  result = 0
  current_entropy = 0
  # weights = data_set["Weights"]
  # print("____________________________")
  # print(data_set)
  b = sum(data_set["Weights"])

  for outcome in outcome_list:
      a = sum(data_set.loc[data_set[label] == outcome, "Weights"])
      # print("A ",a)
      # print("B ",b)
      proportion = a / b
      # print("PROPORTION FOR OUTCOME: ", outcome, " IS ", proportion)
      # print("____________________________")

      current_entropy = -proportion * np.log2(proportion)
      if proportion != 0:
        result += current_entropy
      # adds to the entropy for the whole data set
  # print("Entropy ", result)
  return result

def find_split_attribute(data_set, label, outcome_list):
  attribute_list = data_set.columns.tolist()
  attribute_list.pop() # removes weight from attributes
  attribute_list.pop() # removes label from attributes
  info_gains = []

  for attribute in attribute_list:
    info_gain_val = find_info_gain(data_set,attribute,label,outcome_list)
    info_gains.append(info_gain_val)

  max_gain = max(info_gains)
  split_attribute = attribute_list[info_gains.index(max_gain)]
  # print(split_attribute)
  return split_attribute

def commonLabel(data,label,weights):
  one_sum = sum(data.loc[data[label] == 1, 'Weights'])
  minus_one_sum = sum(data.loc[data[label] == -1, 'Weights'])
  
  if one_sum > minus_one_sum:
    return 1
  else:
    return -1

def Decision_Stump(data, label, outcome_list, attribute_vals):
    root_node = {} # create a root node for the tree
    A = find_split_attribute(data,label, outcome_list)
    root_node[A] = {}
    for v in attribute_vals[A]:
      root_node[A][v] = {} # create a new tree branch for A=v
      S_v = data[data[A] == v] # S_v subset of examples where A = v
      # print(S_v)
      if len(S_v) != 0:
        leaf_node = commonLabel(S_v,label,S_v["Weights"])
       # leaf_node = max(set(S_v[label]), key=list(S_v[label]).count) # leaf node with most common value of Label in S
      else:
        leaf_node = commonLabel(data,label,data["Weights"])#max(set(data[label]), key=list(data[label]).count) # leaf node with most common value of Label in S
      root_node[A][v] = leaf_node
    return root_node

def find_et(decision_stump,data,weights,label):
  test_num = len(data.index)
  e_t = 0
  attribute_to_check = list(decision_stump.keys())[0]
  for row_index in range(test_num):
    row_value = data.iloc[row_index]
    attribute_val = row_value[attribute_to_check]
    tree_res = decision_stump[attribute_to_check][attribute_val]
    if tree_res != data.iloc[row_index][label]: # wrong prediction
      e_t += weights[row_index]
  return e_t

def update_weights(decision_stump,data,weights,label,alpha_t):
  # print("OLD:")
  # print(weights)
  test_num = len(data.index)
  attribute_to_check = list(decision_stump.keys())[0]
  updated_weights = [0] * test_num
  for row_index in range(test_num):
    row_value = data.iloc[row_index]
    attribute_val = row_value[attribute_to_check]
    tree_res = decision_stump[attribute_to_check][attribute_val]
    if tree_res == data.iloc[row_index][label]: # correct prediction
      updated_weights[row_index] = weights[row_index] * np.exp(-alpha_t * 1)
    else: # wrong prediction
      updated_weights[row_index] = weights[row_index] * np.exp(-alpha_t * -1)
  
  # print("NEW:")
  # print([(val / sum(updated_weights)) for val in updated_weights])
  return [(val / sum(updated_weights)) for val in updated_weights]

def compute_vote(e_t):
  # print("E_t is: ", e_t)
  return (0.5 * np.log((1-e_t) / float(e_t)))

def adaBoost(data,label,outcome_list,attribute_vals,t_iter,m):
  D_t = [1/m] * m # Initializing weights
  data["Weights"] = D_t
  weak_classifiers = []
  errors = []
  stump_error = []
  a_t_values = []
  for t in range(t_iter):
      print("Iteration: ", t)
      h_t = Decision_Stump(data,label,outcome_list,attribute_vals)
      e_t = find_et(h_t,data,D_t,label)
      stump_error.append(e_t)
      alpha_t = compute_vote(e_t)
      D_t = update_weights(h_t,data,D_t,label,alpha_t)
      data['Weights'] = D_t
      weak_classifiers.append(h_t)
      a_t_values.append(alpha_t)
      current_error = adaPredict(weak_classifiers,a_t_values,data,label)
      errors.append(current_error)


  return weak_classifiers, a_t_values, errors, stump_error

def adaPredict(weak_classifiers, a_t_values, data, label):
  total_error = 0
  for row_index in range(len(data.index)):
    result = 0
    row_value = data.iloc[row_index]
    for ht in weak_classifiers:
      alphat = a_t_values[weak_classifiers.index(ht)]
      attribute_to_check = list(ht.keys())[0]
      attribute_val = row_value[attribute_to_check]
      expected_output = row_value[label]   
      stump_output = ht[attribute_to_check][attribute_val]
      result += alphat * stump_output
      if expected_output == stump_output:
        result += (alphat * 1)
      else:
        result += (alphat * - 1)
    

    res = np.sign(result)
    if res == -1:
      total_error += 1
    
  return total_error / (len(data.index))     

def prepare_data():

  # data_train = [[41,	'services',	"married",	"secondary",	"no"	,0,	"yes", "no",	"unknown",	5,	"may",	114,	2,	-1,	0,	"unknown",	"no"],
  # [48,	"blue-collar",	"single",	"secondary",	"no",	312,	"yes",	"yes",	"cellular",	3,	"feb",	369,	2,	-1,	0,	"unknown",	"yes"],
  # [55,	"technician",	"married",	"secondary",	"no",	1938,	"no",	"yes",	"cellular",	18,	"aug",	193,	1,	386,	3,	"success",	"no"]]
  df_train = pd.read_csv ('bank_train.csv')

  df_test = pd.read_csv('bank_test.csv')
  df_train.columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign"
  ,"pdays","previous","poutcome","y"]
  att_vals = {}

  df_train.loc[df_train["y"] == "no", "y"] = -1
  df_train.loc[df_train["y"] == "yes", "y"] = 1

  age_median = statistics.median(df_train["age"].tolist())
  df_train.loc[df_train["age"] < age_median, "age"] = 0
  df_train.loc[df_train["age"] >= age_median, "age"] = 1

  balance_median = statistics.median(df_train["balance"].tolist())
  df_train.loc[df_train["balance"] < balance_median, "balance"] = 0
  df_train.loc[df_train["balance"] >= balance_median, "balance"] = 1

  day_median = statistics.median(df_train["day"].tolist())
  df_train.loc[df_train["day"] < day_median, "day"] = 0
  df_train.loc[df_train["day"] >= day_median, "day"] = 1

  duration_median = statistics.median(df_train["duration"].tolist())
  df_train.loc[df_train["duration"] < duration_median, "duration"] = 0
  df_train.loc[df_train["duration"] >= duration_median, "duration"] = 1

  campaign_median = statistics.median(df_train["campaign"].tolist())
  df_train.loc[df_train["campaign"] < campaign_median, "campaign"] = 0
  df_train.loc[df_train["campaign"] >= campaign_median, "campaign"] = 1

  pdays_median = statistics.median(df_train["pdays"].tolist())
  df_train.loc[df_train["pdays"] < pdays_median, "pdays"] = 0
  df_train.loc[df_train["pdays"] >= pdays_median, "pdays"] = 1

  previous_median = statistics.median(df_train["previous"].tolist())
  df_train.loc[df_train["previous"] < previous_median, "previous"] = 0
  df_train.loc[df_train["previous"] >= previous_median, "previous"] = 1

  ### CONVERTED TRAIN DATA TO BINARY ###
  age_median_test = statistics.median(df_test["age"].tolist())
  df_test.loc[df_test["age"] < age_median_test, "age"] = 0
  df_test.loc[df_test["age"] >= age_median_test, "age"] = 1

  balance_median_test = statistics.median(df_test["balance"].tolist())
  df_test.loc[df_test["balance"] < balance_median_test, "balance"] = 0
  df_test.loc[df_test["balance"] >= balance_median_test, "balance"] = 1

  day_median_test = statistics.median(df_test["day"].tolist())
  df_test.loc[df_test["day"] < day_median_test, "day"] = 0
  df_test.loc[df_test["day"] >= day_median_test, "day"] = 1

  duration_median_test = statistics.median(df_test["duration"].tolist())
  df_test.loc[df_test["duration"] < duration_median_test, "duration"] = 0
  df_test.loc[df_test["duration"] >= duration_median_test, "duration"] = 1

  campaign_median_test = statistics.median(df_test["campaign"].tolist())
  df_test.loc[df_test["campaign"] < campaign_median_test, "campaign"] = 0
  df_test.loc[df_test["campaign"] >= campaign_median_test, "campaign"] = 1

  pdays_median_test = statistics.median(df_test["pdays"].tolist())
  df_test.loc[df_test["pdays"] < pdays_median_test, "pdays"] = 0
  df_test.loc[df_test["pdays"] >= pdays_median_test, "pdays"] = 1

  previous_median_test = statistics.median(df_test["previous"].tolist())
  df_test.loc[df_test["previous"] < previous_median_test, "previous"] = 0
  df_test.loc[df_test["previous"] >= previous_median_test, "previous"] = 1


  df_test.loc[df_test["y"] == "no", "y"] = -1
  df_test.loc[df_test["y"] == "yes", "y"] = 1


  ### CONVERTED TEST DATA TO BINARY
  att_vals['age'] = [0,1]
  att_vals['job'] = df_train['job'].unique().tolist()
  att_vals['marital'] = ["married","divorced","single"]
  att_vals['education'] = ["unknown", "secondary","primary","tertiary"]
  att_vals['default'] = ['yes','no']
  att_vals['balance'] = [0,1]
  att_vals['housing'] = ['yes','no']
  att_vals['loan'] = ['yes','no']
  att_vals['contact'] = ['unknown','telephone','cellular']
  att_vals['day'] = [0,1]
  att_vals['month'] = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
  att_vals['duration'] = [0,1]
  att_vals['campaign'] = [0,1]
  att_vals['pdays'] = [0,1]
  att_vals['previous'] = [0,1]
  att_vals['poutcome'] = ['unknown','other','failure','success']
  att_vals['y'] = [-1,1]

  return df_train, df_test, att_vals



  # attribute_to_check = list(decision_tree.keys())[0]
  # updated_weights = [0] * test_num
  # for row_index in range(test_num):
  #   print(row_index)
  #   row_value = test_data.iloc[row_index]
  #   attribute_val = row_value[attribute_to_check]
  #   tree_res = decision_tree[attribute_to_check][attribute_val]
  #   while type(tree_res) == dict:
  #     next_attribute_to_check = list(tree_res.keys())[0]
  #     next_attribute_val = row_value[next_attribute_to_check]
  #     tree_res = tree_res[next_attribute_to_check][next_attribute_val]
  
  #     if tree_res != test_data.iloc[row_index][label]: # correct prediction

def test():
  df_train, df_test, att_vals = prepare_data()

  # Running AdaBoost
  classifiers_train, ats_train,errors_train ,stump_errors_train = adaBoost(df_train,"y",
  [-1,1],att_vals,500,len(df_train.index))
  
  classifiers_test, ats_test, errors_test, stump_errors_test = adaBoost(df_test,"y",
  [-1,1],att_vals,500,len(df_test.index))

  plt.plot(errors_train,np.arange(500),color='r', label = 'Training Data')
  plt.plot(errors_test,np.arange(500),color='g', label = 'Testing Data')
  plt.xlabel("Error")
  plt.ylabel("T")
  plt.legend()
  plt.show()
  plt.plot(stump_errors_train,np.arange(500),color='r', label = 'Training Data')
  plt.plot(stump_errors_test,np.arange(500),color='g', label = 'Testing Data')
  plt.xlabel("Decision Stump Error")
  plt.ylabel("T")
  plt.legend()
  plt.show()
  # print(classifiers,ats)
 


test()
