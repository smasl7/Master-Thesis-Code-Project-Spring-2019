import numpy as np
import pandas as pd
from html import unescape
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction

folder_dir = "./"
data_dir_drugs = folder_dir + "drugsCom_raw/"

def print_data_stats(data, msg):
  print(msg)
  print("Size of the data set:", data.shape)
  print("Number of unique drugs:", len(set(data[:,0])))
  c_drug = Counter(data[:,0])
  print("Top 15 most frequent drugs (and their count):", c_drug.most_common(15))
  print("Number of unique conditions:", len(set(data[:,1])))
  c_cond = Counter(data[:,1])
  print("Top 15 most frequent conditions (and their count):", c_cond.most_common(15))  
  print("Stats around number of characters in the data set")
  lengths = [len(x) for x in data[:,2]]
  print("- Maximum number of characters in a drug review:", np.max(lengths))
  print("- Minimum number of characters in a drug review:", np.min(lengths))
  print("- Average number of characters in a drug review:", np.mean(lengths))
  print("- Standard deviation aruond number of characters in a drug review:", np.std(lengths))

def is_valid(sample):
  no_parsing_error_or_empty = not "</span>" in sample[1] if isinstance(sample[1], str) else sample[1]!=-1
  if not no_parsing_error_or_empty:
    return False

  connected_with_review = sample[0] in sample[2] or sample[1] in sample[2] or "".join([x[0] for x in sample[1].split()]) in sample[2]
  if not connected_with_review:
    return False
  
  acceptable_size_of_drug_review = 400<=len(sample[2]) and len(sample[2])<=500
  if not acceptable_size_of_drug_review:
    return False
  
  satisfying_usefulcount_threshold = 10<=sample[5]
  if not satisfying_usefulcount_threshold:
    return False
  
  return True

def duplicate_free(data):
  present = defaultdict(lambda: (False, -1))
  new_data = []
  
  for entry in data:
    key = (entry[1], entry[2], entry[3])
    if (not present[key][0]) and entry[0] in entry[2]:
      new_data.append(entry)
      present[key] = (True, len(new_data) - 1)
    elif entry[0] in entry[2]:
      if new_data[present[key][1]][0] in entry[2]:
        new_data.append(entry)
      else:
        new_data[present[key][1]][0] = entry[0]
  
  return np.array(new_data)

def free_from_infrequents(data, min_count = 10):
  drug_counts = Counter(data[:,0])
  cond_counts = Counter(data[:,1])
  
  new_data = []
  for entry in data:
    if drug_counts[entry[0]]>=min_count and cond_counts[entry[1]]>=min_count:
      new_data.append(entry)
    
  return np.array(new_data)

def get_bleu_score(review, test_data, rating_span = 5):
  try:
    candidate = review[2].split()
    references = [ref_review[2].split() for ref_review in test_data if (ref_review[[1,3]]==review[[1,3]]).all()]
    return bleu(references, candidate, smoothing_function = SmoothingFunction().method4)
  except KeyError as e:
    print("WARNING: No reference translations were obtainable for the following review:", review)
    return -1