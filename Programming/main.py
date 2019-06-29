"""
Author: Sylwester Liljegren
Date: 2019-06-28

This is the main file that runs all the experiments to produce the results that were presented in the official report 
of the conducted master thesis.

In this main file, the codes behind the experiments are visible to the public and to produce other results using 
other experimental parameters, one could go down below a bit in this file to edit those parameters.
"""

### LIBRARIES ###
import numpy as np # NumPy library
from utils import * # All functions defined in utils.py
from lstm_model import * # The implementation of the LSTM model
from nn_model import * # The implementation of the NN model
from v_att2seq_model import * # The implementation of the V-Att2Seq model
import pandas as pd # Pandas library
import os
from sklearn.model_selection import train_test_split # SciKit-Lean implementation for randomly splitting data sets into two partitions
from collections import Counter # Counter from collections used to keep counter of # of times a distinct element occurs within a data 
# structure


folder_dir = "./" # Current folder
drugs_data_dir = folder_dir + "drugsCom_raw/" # The folder for the used data sets
drugs_train_data_name = drugs_data_dir + "drugsComTrain_" # The (absolute path) name to training data set
drugs_test_data_name = drugs_data_dir + "drugsComTest_" # The (absolute path) name to testing data set
RAW = "raw" # Keyword for the data set(s) being un-preprocessed
PREPROCESSED = "preprocessed" # Keyword for the data set(s) being preprocessed
file_end = ".tsv" # File extension for the data sets
update = False # Pre-process (i.e. update) the data set(s) from the beginning in the upcoming execution
MAX_REVIEW_LENGTH = 175 # Maximum length of the drug reviews in relevant data sets
MIN_REVIEW_LENGTH = 15 # Minimum length of the drug reviews in relevant data sets
MIN_GROUP_SIZE = 24 # Minimum number of attribute sets that are present in the data set(s)
FRAC = 0.7 # Fraction of the entire data set used as training data (the rest used as testing data set)
REPS = 10 # Number of repetitions of the experiments to perform with each concerned model


"""
This function converts all occurring attribute sets on the form (drug, condition, rating) in the data set <data> 
to some integer. Reason for this conversion is to enable stratification procedudure that is offered by the 
SciKit-Learn library to maintain approximately equivalent class distributions between the training- and testing 
data set.
"""
def attrs_to_ints(data):
  keys = list(set(map(tuple, data[:, [0, 1, 3]]))) # Retrieve all unique attribute sets from <data> on the form (drug, condition, rating)
  to_int = {} # Initialize the dictionary mapping an attribute set to some integer as empty

  # Add mappings between the unique attribute sets and integers
  for i, key in enumerate(keys): # For each unique attribute set
    to_int[key] = i # Assign an integer to that attribute set
  
  # Go through the entire data set and obtain the integers of the samples given its attribute set
  res = [] # Initialize list as empty
  for entry in data: # For each sample in <data>
    key = tuple(entry[[0, 1, 3]]) # Get attribute set
    res.append(to_int[key]) # Obtain the integer from the given attribute set and append it to the list
  
  return np.array(res) # Return the NumPy array version of the appended list


"""
main() defines the entire course for conducting all the relevant experiments, beginning from loading the 
relevant data sets until obtaining the wished results that were later used in the report.

To call this program, one could either run it directly in some IDE of own preference that has Python 
interpreter, or one could simply call "python main.py" to run this program in e.g. a shell.
"""
def main():
  """
  Check whether there are preprocessed versions of the relevant data sets and whether an update is not requested (by <update> value)
  """
  # If there are preprocessed versions of the relevant data sets and no update is requested
  if os.path.exists(drugs_train_data_name + PREPROCESSED + file_end) and os.path.exists(drugs_test_data_name + PREPROCESSED + file_end) and not update:
    print("Loading preprocessed data sets...")
    # Load the data sets using Pandas, where all the index columns are removed, NaN's are replaced with -1 and all alphabetic characters 
    # are lowered using *.lower()
    training = np.array(pd.read_csv(drugs_train_data_name + PREPROCESSED + file_end, sep = "\t").fillna(-1).applymap(lambda s: s.lower() if isinstance(s, str) else str(s)))[:,1:]
    testing = np.array(pd.read_csv(drugs_test_data_name + PREPROCESSED + file_end, sep = "\t").fillna(-1).applymap(lambda s: s.lower() if isinstance(s, str) else str(s)))[:,1:]
    print("Done with loading preprocessed data sets!")
  else:
    # If there are no preprocessed versions of the relevant data sets and/or an update is requested
    print("Loading raw data sets...")
    # Load the data sets using Pandas, where all the index columns are removed, NaN's are replaced with -1 and all alphabetic characters 
    # are lowered using *.lower()
    training = np.array(pd.read_csv(drugs_train_data_name + RAW + file_end, sep = "\t").fillna(-1).applymap(lambda s: s.lower() if isinstance(s, str) else str(s)))[:,1:]
    testing = np.array(pd.read_csv(drugs_test_data_name + RAW + file_end, sep = "\t").fillna(-1).applymap(lambda s: s.lower() if isinstance(s, str) else str(s)))[:,1:]  
    print("Done with loading raw data sets!")

    """
    Training data set
    """
    print("Preprocessing the training data set...")
    training[:,2] = np.array(list(map(format_text, training[:,2]))) # Modify the texts using the format_text(...) function as defined in utils.py
    print(training[:15]) # Get some examples of the samples after applying the format_text(...) to the data set
    training = np.array([sample for sample in training if is_valid(sample)]) # Remove all samples that are not wanted for the experiments
    training[:,2] = np.array(list(map(correctly_spelled, training))) # Correct the spelling of the drug reviews in each sample of the data set
    # using the correctly_spelled(...) function as defined in utils.py
    print(training[:15]) # Get some examples of the samples after applying the correctly_spelled(...) function
    print("Done with preprocessing the training data set!")

    """
    Testing data set
    """
    print("Preprocessing the testing data set...")
    testing[:,2] = np.array(list(map(format_text, testing[:,2]))) # Modify the texts using the format_text(...) function as defined in utils.py
    print(testing[:15]) # Get some examples of the samples after applying the format_text(...) to the data set
    testing = np.array([sample for sample in testing if is_valid(sample)]) # Remove all samples that are not wanted for the experiments
    testing[:,2] = np.array(list(map(correctly_spelled, testing))) # Correct the spelling of the drug reviews in each sample of the data set
    # using the correctly_spelled(...) function as defined in utils.py
    print(testing[:15]) # Get some examples of the samples after applying the correctly_spelled(...) function
    print("Done with preprocessing the testing data set!")
    
    print("Saving the preprocessed data sets...")
    pd.DataFrame(training).to_csv(drugs_train_data_name + PREPROCESSED + file_end, sep = "\t") # Save the preprocessed training data set into a .tsv file
    pd.DataFrame(testing).to_csv(drugs_test_data_name + PREPROCESSED + file_end, sep = "\t") # Save the preprocessed testing data set into a .tsv file
    print("Done with saving the preprocessed data sets!")
    print("- Training data set saved at following path:", os.path.join(os.path.dirname(drugs_train_data_name + PREPROCESSED + file_end), drugs_train_data_name + PREPROCESSED + file_end))
    print("- Testing data set saved at following path:", os.path.join(os.path.dirname(drugs_test_data_name + PREPROCESSED + file_end), drugs_test_data_name + PREPROCESSED + file_end), end = "\n\n")
  
  data = np.vstack((training, testing))[:,:-2] # Merge the training- and testing data sets together
  data = free_from_infrequents(np.array([e for e in data if MIN_REVIEW_LENGTH<=len(e[2]) and len(e[2])<=MAX_REVIEW_LENGTH]), MIN_GROUP_SIZE) # Remove samples containing inappropriate lengths on drug 
  # review and remove the samples having an attribute set not occurring frequently (provided the threshold <MIN_GROUP_SIZE>)
  print_data_stats(data, "##### Basic stats about the data set w.r.t. drugs and diseases #####")

  # Initialize the lists
  lstm_bleus = [] # List for containing BLEU scores of the LSTM model on the form (BLEU_train, BLEU_test)
  v_att2seq_bleus = [] # List for containing the BLEU scores of the V-Att2Seq model on the form (BLEU_train, BLEU_test)
  nn_bleus = [] # List for containing the BLEU scores of the NN model on the form (BLEU_train, BLEU_test)
  attrs_to_classes = attrs_to_ints(data) # A NumPy array of the integers converted from the attribute sets (for 
  # programmatically enabling stratification procedure)

  print("########## CONDUCTING THE EXPERIMENTS NOW! ##########")
  for i in range(REPS): # For each repetition
    print("---------- Repetition #" + str(i+1) + " ----------")    
    # Split randomly the data into training- and testing data sets in a stratified manner,
    # where 100*<FRAC> % is used as training data set and the rest as testing data set
    train, test = train_test_split(data, train_size = FRAC, stratify = attrs_to_classes, shuffle = True)

    """
    LSTM model
    """
    lstm_model = LSTM_model() # Initialization with default parameters
    lstm_model.fit(train[:,2]) # Fit the model with the drug reviews in the training data set
    print("Evaluating the LSTM model...", end = " ")
    train_lstm_bleu_score = evaluate_bleu_score(lstm_model, train, MAX_REVIEW_LENGTH) # Calculate the BLEU score on the training data set
    test_lstm_bleu_score = evaluate_bleu_score(lstm_model, test, MAX_REVIEW_LENGTH) # Calculate the BLEU score on the testing data set
    print("Done!")
    print("BLEU score on training data by LSTM model:", np.round(train_lstm_bleu_score, 2))
    print("BLEU score on testing data by LSTM model:", np.round(test_lstm_bleu_score, 2))
    print("-----------------------------------------------------------------")
    lstm_bleus.append((train_lstm_bleu_score, test_lstm_bleu_score)) # Append the list with BLEU scores of the LSTM model with the recent BLEU scores
    
    """
    NN model
    """
    nn_model = NN_model() # Initialization with default parameters
    nn_model.fit(train) # Fit the model with the training data set
    print("Evaluating the NN model...", end = " ")
    train_NN_bleu_score = evaluate_bleu_score(nn_model, train, MAX_REVIEW_LENGTH) # Calculate the BLEU score on the training data set
    test_NN_bleu_score = evaluate_bleu_score(nn_model, test, MAX_REVIEW_LENGTH) # Calculate the BLEU score on the testing data set
    print("Done!")
    print("BLEU score on training data by NN model:", np.round(train_NN_bleu_score, 2))
    print("BLEU score on testing data by NN model:", np.round(test_NN_bleu_score, 2))
    print("-----------------------------------------------------------------")
    nn_bleus.append((train_NN_bleu_score, test_NN_bleu_score)) # Append the list with BLEU scores of the NN model with the recent BLEU scores
  
    """
    V-Att2Seq model
    """
    vatt2seq_model = V_Att2Seq_model() # Initialization with default parameters
    vatt2seq_model.fit(train) # Fit the model with the training data set
    print("Evaluating the V-Att2Seq model...", end = " ")
    train_v_att2seq_bleu_score = evaluate_bleu_score(vatt2seq_model, train, MAX_REVIEW_LENGTH) # Calculate the BLEU score on the training data set
    test_v_att2seq_bleu_score = evaluate_bleu_score(vatt2seq_model, test, MAX_REVIEW_LENGTH) # Calculate the BLEU score on the testing data set
    print("Done!")
    print("BLEU score on training data by V-Att2Seq model:", np.round(train_v_att2seq_bleu_score, 2))
    print("BLEU score on testing data by V-Att2Seq model:", np.round(test_v_att2seq_bleu_score, 2))
    print("-----------------------------------------------------------------")
    v_att2seq_bleus.append((train_v_att2seq_bleu_score, test_v_att2seq_bleu_score)) ## # Append the list with BLEU scores of the V-Att2Seq model with the recent BLEU scores

    # Store the BLEU scores from all the above models into a text file named "exp_res.txt" (note: append mode)
    with open("exp_res.txt", "a") as f:
      f.write("----- Rep #" + str(i+1) + " -----\n")
      f.write("LSTM model: (Train) " + str(train_lstm_bleu_score) + " - (Test) " + str(test_lstm_bleu_score) + "\n")
      f.write("NN model: (Train) " + str(train_NN_bleu_score) +  " - (Test) " + str(test_NN_bleu_score) + "\n")
      f.write("V-Att2Seq model: (Train) " + str(train_v_att2seq_bleu_score) + " - (Test) " + str(test_v_att2seq_bleu_score) + "\n")

    if i<REPS-1:
      print("**********************************************************")  

  print("########## DONE WITH CONDUCTING THE EXPERIMENTS ##########", end = "\n\n")

  print("########## FINAL RESULTS FROM THE EXPERIMENTS ##########")
  print("BLEU scores of the LSTM model:", lstm_bleus)
  print("BLEU scores of the NN model:", nn_bleus)
  print("BLEU scores of the V-Att2Seq model:", v_att2seq_bleus, end = "\n\n")


# Run the main file if it is specified to be the main file
if __name__ == "__main__":
  main()