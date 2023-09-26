import pandas as pd
import numpy as np
from collections import Counter
import warnings
import statistics
warnings.filterwarnings("ignore")

def find_Gini(data_set, label, outcome_list):
  total_count = len(data_set.index)
  prop_square_sum = 0
  for outcome in outcome_list:
    outcome_count = len(data_set[data_set[label]==outcome])
    outcome_proportion = outcome_count / total_count
    prop_square_sum += (outcome_proportion * outcome_proportion)

  return 1 - prop_square_sum

def info_gain_Gini(data_set, attribute, label, outcome_list):
  attribute_vals = data_set[attribute].unique()
  current_info = 0.0
  total_data_count = len(data_set.index)
  total_Gini = find_Gini(data_set,label,outcome_list)

  for val in attribute_vals:
    df_sorted = data_set[data_set[attribute] == val]
    sorted_data_count = len(df_sorted.index)
    subset_gini = find_Gini(df_sorted,label,outcome_list)
    attribute_proportion = sorted_data_count / total_data_count
   # print(attribute, " entropy is " ,  subset_gini)
    if pd.isna(subset_gini) == False:
      current_info += attribute_proportion * subset_gini


  return total_Gini - current_info

def find_split_attribute_Gini(data_set, label, outcome_list):
  attribute_list = data_set.columns.drop(label)
  info_gains = []

  for attribute in attribute_list:
    info_gain_val = info_gain_Gini(data_set,attribute,label,outcome_list)
    info_gains.append(info_gain_val)

  max_gain = max(info_gains)
  split_attribute = attribute_list[info_gains.index(max_gain)]

  return split_attribute

def Id3_GINI(data, attributes, label, outcome_list, attribute_vals, tree, max_depth, current_depth):
  if tree is None:
      tree = {}

  if current_depth == max_depth:
    leaf_node = max(set(data[label]), key=list(data[label]).count) 
    return leaf_node

  if len(set(data[label])) == 1: # all examples have same label
    if len(attributes) == 0: 
      leaf = max(set(data[label]), key=data[label].count) 
     # leaf node with most common value
    else:
        # print(data[label])
        leaf = data[label].unique()[0]
      # leaf node with the only value
    return leaf
  else:
    root_node = {} # create a root node for the tree
    A = find_split_attribute_Gini(data, label, outcome_list)
    root_node[A] = {}
    for v in attribute_vals[A]:
      root_node[A][v] = {} # create a new tree branch for A=v
      S_v = data[data[A] == v] # S_v subset of examples where A = v
      if len(S_v) == 0: # if S_v is empty
        leaf_node = max(set(data[label]), key=list(data[label]).count) # leaf node with most common value of Label in S
        root_node[A][v] = leaf_node
      else:      
          root_node[A][v] = Id3_IG(S_v, attributes, label, outcome_list, attribute_vals,tree, max_depth, current_depth + 1)
    return root_node


############## END GINI INDEX ####################



def find_info_gain(data_set, attribute, label, outcome_list):
  attribute_vals = data_set[attribute].unique()
  current_info = 0.0
  total_data_count = len(data_set.index)
  total_entropy = find_data_entropy(data_set,label,outcome_list)

  for val in attribute_vals:
    df_sorted = data_set[data_set[attribute] == val]
    sorted_data_count = len(df_sorted.index)
    ent = find_data_entropy(df_sorted,label,outcome_list)
    attribute_proportion = sorted_data_count / total_data_count
   # print(attribute, " entropy is " ,  ent)
    if pd.isna(ent) == False:
      current_info += attribute_proportion * ent


  return total_entropy - current_info

def find_data_entropy(data_set, label, outcome_list):
    # data_set -> data set that the entropy will be calculated for.
    # label -> attribute that the outcome is displayed under 'y' for HW1 Q1.
    # outcome_list -> possible outcomes that can appear under the label. [0,1] for HW1 Q1.
    result = 0
    current_entropy = 0
    total_data_count = len(data_set.index) # finds the number of data in dataset to use it while calculating proprotion

    for outcome in outcome_list:
        proportion = len(data_set[data_set[label]==outcome]) / total_data_count
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

def Id3_IG(data, attributes, label, outcome_list, attribute_vals, tree, max_depth, current_depth):
  if tree is None:
      tree = {}

  if current_depth == max_depth:
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
          root_node[A][v] = Id3_IG(S_v, attributes, label, outcome_list, attribute_vals,tree, max_depth, current_depth + 1)
    return root_node

############## END INFO GAIN #############################
def find_ME(data_set,label, outcome_list):
  total_count = len(data_set.index)
  count_list = []
  for outcome in outcome_list:
    outcome_count = len(data_set[data_set[label] == outcome])
    count_list.append(outcome_count)

  majority_count = max(count_list)
  count_list.remove(majority_count)

  return sum(count_list) / total_count

def info_gain_ME(data_set, attribute, label, outcome_list):
  attribute_vals = data_set[attribute].unique()
  current_info = 0.0
  total_data_count = len(data_set.index)
  total_ME = find_ME(data_set,label,outcome_list)

  for val in attribute_vals:
    df_sorted = data_set[data_set[attribute] == val]
    sorted_data_count = len(df_sorted.index)
    m_error = find_ME(df_sorted,label,outcome_list)
    attribute_proportion = sorted_data_count / total_data_count
    if pd.isna(m_error) == False:
      current_info += attribute_proportion * m_error


  return total_ME - current_info

def find_split_attribute_ME(data_set, label, outcome_list):
  attribute_list = data_set.columns.drop(label)
  info_gains = []

  for attribute in attribute_list:
    info_gain_val = info_gain_ME(data_set,attribute,label,outcome_list)
    info_gains.append(info_gain_val)

  max_gain = max(info_gains)
  split_attribute = attribute_list[info_gains.index(max_gain)]

  return split_attribute

def Id3_ME(data, attributes, label, outcome_list, attribute_vals, tree, max_depth, current_depth):
  if tree is None:
      tree = {}

  if current_depth == max_depth:
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
    A = find_split_attribute_ME(data, label, outcome_list)
    root_node[A] = {}
    for v in attribute_vals[A]:
      root_node[A][v] = {} # create a new tree branch for A=v
      S_v = data[data[A] == v] # S_v subset of examples where A = v
      if len(S_v) == 0: # if S_v is empty
        leaf_node = max(set(data[label]), key=list(data[label]).count) # leaf node with most common value of Label in S
        root_node[A][v] = leaf_node
      else:      
          root_node[A][v] = Id3_IG(S_v, attributes, label, outcome_list, attribute_vals,tree, max_depth, current_depth + 1)
    return root_node


############ END MAJORITY ERROR #######################

def predict_data(decision_tree, test_data, dtype):
  num_right_predictions = 0
  num_wrong_predictions = 0
  test_num = len(test_data.index)
  if dtype == "Car":
    label = 'label'
  if dtype == "Bank":
    label = 'y'
  attribute_to_check = list(decision_tree.keys())[0]
  
  for row_index in range(test_num):
    row_value = test_data.iloc[row_index]
    attribute_val = row_value[attribute_to_check]
    tree_res = decision_tree[attribute_to_check][attribute_val]
    while type(tree_res) == dict:
      next_attribute_to_check = list(tree_res.keys())[0]
      next_attribute_val = row_value[next_attribute_to_check]
      tree_res = tree_res[next_attribute_to_check][next_attribute_val]
    
    if tree_res == test_data.iloc[row_index][label]: # correct prediction
      num_right_predictions += 1
    else: # wrong prediction
      num_wrong_predictions += 1
  
  error = num_wrong_predictions / test_num

  return error

def q2():
  print("---------- QUESTION 2b ----------")
  ### CAR ###
  df_train = pd.read_csv ('train.csv')
  df_test = pd.read_csv('test.csv')
  df_train.columns = ["buying","maint","doors","persons","lug_boot","safety","label"]
  att_vals = {}
  att_vals['buying'] = ['vhigh','high','med','low']
  att_vals['maint'] = ['vhigh','high','med','low']
  att_vals['doors'] = ['2','3','4','5more']
  att_vals['persons'] = ['2','4','more']
  att_vals['lug_boot'] = ['small','med','big']
  att_vals['safety'] = ['low','med','high']

  test_tech = ["ME", "IG", "GINI"]
  print("TRAINING DATA")
  for tech in test_tech:
    test_res = []
    sum_err = 0
    for i in range(6):
      if tech == "IG":
        t_test = Id3_IG(df_train,["buying","maint","doors","persons","lug_boot","safety"],"label",['unacc','acc', 'good', 'vgood'], att_vals,None,i+1,0)
      elif tech == "ME":
        t_test = Id3_ME(df_train,["buying","maint","doors","persons","lug_boot","safety"],"label",['unacc','acc', 'good', 'vgood'], att_vals,None,i+1,0)
      else:
        t_test = Id3_GINI(df_train,["buying","maint","doors","persons","lug_boot","safety"],"label",['unacc','acc', 'good', 'vgood'], att_vals,None,i+1,0)

      test_res.append(predict_data(t_test,df_train,"Car"))
      sum_err += predict_data(t_test,df_train, "Car")
    print("END OF ", tech, "RESULT ARRAY IS: ", test_res)
    print("END OF ", tech, "AVERAGE ERROR IS: ", sum_err / 6)
    print("--------------------------------------------")
      
  print("TESTING DATA")
  for tech in test_tech:
    test_res = []
    sum_err = 0
    for i in range(6):
      if tech == "IG":
        t_test = Id3_IG(df_train,["buying","maint","doors","persons","lug_boot","safety"],"label",['unacc','acc', 'good', 'vgood'], att_vals,None,i+1,0)
      elif tech == "ME":
        t_test = Id3_ME(df_train,["buying","maint","doors","persons","lug_boot","safety"],"label",['unacc','acc', 'good', 'vgood'], att_vals,None,i+1,0)
      else:
        t_test = Id3_GINI(df_train,["buying","maint","doors","persons","lug_boot","safety"],"label",['unacc','acc', 'good', 'vgood'], att_vals,None,i+1,0)

      test_res.append(predict_data(t_test,df_test,"Car"))
      sum_err += predict_data(t_test,df_test, "Car")
    print("END OF ", tech, "RESULT ARRAY IS: ", test_res)
    print("END OF ", tech, "AVERAGE ERROR IS: ", sum_err / 6)
    print("--------------------------------------------")

def q3():
  ### BANK ###
  print("---------- QUESTION 3a ----------")
  df_train = pd.read_csv ('bank_train.csv')
  df_test = pd.read_csv('bank_test.csv')
  df_train.columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign"
  ,"pdays","previous","poutcome","y"]
  att_vals = {}

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
  att_vals['y'] = ['yes','no']

  test_tech = ["ME", "IG", "GINI"]
  print("TRAINING DATA")
  for tech in test_tech:
    test_res = []
    sum_err = 0
    for i in range(16):
      if tech == "IG":
        t_test = Id3_IG(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      elif tech == "ME":
        t_test = Id3_ME(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      else:
        t_test = Id3_GINI(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      
      test_res.append(predict_data(t_test,df_train,"Bank"))
      sum_err += predict_data(t_test,df_train, "Bank")
    print("END OF ", tech, "RESULT ARRAY IS: ", test_res)
    print("END OF ", tech, "AVERAGE ERROR IS: ", sum_err / 16)
    print("--------------------------------------------")
        
  print("TESTING DATA")
  for tech in test_tech:
    test_res = []
    sum_err = 0
    for i in range(16):
      if tech == "IG":
        t_test = Id3_IG(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      elif tech == "ME":
        t_test = Id3_ME(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      else:
        t_test = Id3_GINI(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)

      test_res.append(predict_data(t_test,df_test,"Bank"))
      sum_err += predict_data(t_test,df_test, "Bank")
    print("END OF ", tech, "RESULT ARRAY IS: ", test_res)
    print("END OF ", tech, "AVERAGE ERROR IS: ", sum_err / 16)
    print("--------------------------------------------")

def q3b():
  ### BANK ###
  print("---------- QUESTION 3b ----------")
  df_train = pd.read_csv ('bank_train.csv')
  df_test = pd.read_csv('bank_test.csv')
  df_train.columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign"
  ,"pdays","previous","poutcome","y"]
  att_vals = {}

  for att in df_train.columns:
    common_item = df_train[att].value_counts().idxmax()
    df_train.loc[df_train[att] == "unknown", att] = common_item

  for att in df_train.columns:
    common_item = df_test[att].value_counts().idxmax()
    df_test.loc[df_test[att] == "unknown", att] = common_item


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
  att_vals['y'] = ['yes','no']

  test_tech = ["ME", "IG", "GINI"]
  print("TRAINING DATA")
  for tech in test_tech:
    test_res = []
    sum_err = 0
    for i in range(16):
      if tech == "IG":
        t_test = Id3_IG(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      elif tech == "ME":
        t_test = Id3_ME(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      else:
        t_test = Id3_GINI(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      
      test_res.append(predict_data(t_test,df_train,"Bank"))
      sum_err += predict_data(t_test,df_train, "Bank")
    print("END OF ", tech, "RESULT ARRAY IS: ", test_res)
    print("END OF ", tech, "AVERAGE ERROR IS: ", sum_err / 16)
    print("--------------------------------------------")
        
  print("TESTING DATA")
  for tech in test_tech:
    test_res = []
    sum_err = 0
    for i in range(16):
      if tech == "IG":
        t_test = Id3_IG(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      elif tech == "ME":
        t_test = Id3_ME(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)
      else:
        t_test = Id3_GINI(df_train,df_train.columns,"y", ['yes','no'], att_vals,None,i+1,0)

      test_res.append(predict_data(t_test,df_test,"Bank"))
      sum_err += predict_data(t_test,df_test, "Bank")
    print("END OF ", tech, "RESULT ARRAY IS: ", test_res)
    print("END OF ", tech, "AVERAGE ERROR IS: ", sum_err / 16)
    print("--------------------------------------------")

def print_answers():
  q2()
  q3()
  q3b()

print_answers()



    




  






