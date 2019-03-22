# -*- coding: utf-8 -*-
import numpy as np
from html import unescape
from baseline_interface import *
from suggested_alternative_interface import *
from dong_interface import *
from hu_interface import *
from utils import *

def main():
  training = np.array(pd.read_csv(data_dir_drugs + "drugsComTrain_raw.tsv", sep = "\t").fillna(-1).applymap(lambda s: s.lower() if isinstance(s, str) else s))[:,1:]  # Remove the columns containing the indices for the drug reviews
  testing = np.array(pd.read_csv(data_dir_drugs + "drugsComTest_raw.tsv", sep = "\t").fillna(-1).applymap(lambda s: s.lower() if isinstance(s, str) else s))[:,1:]  # Remove the columns containing the indices for the drug reviews

  data_before = np.vstack((training, testing))
  print_data_stats(data_before, "##### Basic stats about the data set w.r.t. drugs and diseases #####")

  training[:,2] = np.array([unescape(sample[1:-1]) for sample in training[:,2]])
  training = free_from_infrequents(duplicate_free(np.array([sample for sample in training if is_valid(sample)])))
  testing[:,2] = np.array([unescape(sample[1:-1]) for sample in testing[:,2]])
  testing = free_from_infrequents(duplicate_free(np.array([sample for sample in testing if is_valid(sample)])))
  print_data_stats(training, "##### Basic stats about the training data set w.r.t. drugs and diseases #####")
  print(training, end = "\n\n")
  print_data_stats(testing, "##### Basic stats about the testing data set w.r.t. drugs and diseases #####")
  print(testing, end = "\n\n")

  # Retrieve drug reviews from the data set
  baseline_train = np.array([sample for sample in training if 
                            is_valid_for_baseline(sample)])[:,2]
  baseline_test = np.array([sample for sample in testing if 
                            is_valid_for_baseline(sample, False)])[:,2]

  print(baseline_train.shape)
  # Example commands to both train- and run the above implementation
  baseline_model = LSTM_model(word_level = False, embedding_size = 600, num_units = 250, drop_out = 0)
  baseline_model.fit(baseline_train, 1)
  review = np.random.choice(baseline_train)
  print(review)
  print(baseline_model.generate((review.split())[:5], len((review.split())[5:]), 0.15))

  """
  sugg_train = np.array([sample for sample in training if is_valid_for_suggestion(sample)])[:,:-2]
  sugg_test = np.array([sample for sample in testing if is_valid_for_suggestion(sample, False)])[:,:-2]
  
  print(sugg_train.shape)
  # Example commands to both train- and run the above implementation
  suggested_model = suggested_alternative(True)
  suggested_model.fit(sugg_train)
  review = np.random.choice(sugg_train)
  print(review)
  print(suggested_model.generate(review[:15], len(review[15:])))

  # Conduction of the experiments of each model
  i = np.random.randint(testing.shape[0])
  print(get_bleu_score(testing[i], testing[[j for j in range(testing.shape[0]) if j!=i]]))
  """

main()