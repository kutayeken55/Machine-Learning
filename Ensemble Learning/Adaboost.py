# Importing functions from MLTEST.py to call Decision Tree Algorithms
import sys
sys.path.append("DecisionTree")
import MLTEST.py
# We have 'm' examples
# D_t is a set of weights over the examples [D_t(1),...., D_t(m)]
# initially uniform dist. weight is 1/m
# Compute Info Gain to select the best feature

# New weight  = sample weight * e^{amount of say} while increasing sample weight for incorrectly classified examples
# New weight = sample weight * e^{- amount of say} while decreasing sample weight for correctly classified examples.
# Amount of say  = 0.5 * log((1- Total Error) / Total Error)
# NORMALIZE VALUES THEN
def adaBoost(t_iter,m):
    D_t = [1/m] * m # Initializing weights
    for t in range(t_iter):
        # find a classifier h_t whose weighted classification error is better
        # compute its vote (amount of say):
        # alpha_t = 0.5 ln ((1 - e_t) / e_t)
        # Update the values of the weights for the training examples.
    
    # Return the final hypothesis,
        
        
