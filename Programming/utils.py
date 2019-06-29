"""
Author: Sylwester Liljegren
Date: 2019-06-29

This file contains all the utility functions that are used to perform each specific task that is relevant for running the experiments,
starting from loading (and additionally preprocessing) data sets to obtain BLEU scores of the relevant models. All the functions are 
used across different files that are included in this code project. This mostly refers to the main file (main.py), but also other 
files such as lstm_model.py and v_att2seq_model.py.
"""

### LIBRARIES ###
import numpy as np # NumPy library
import os
import contractions # PyContractions used for extending contractions in the English language
import re # Regex library in Python
from html import unescape # unescape(...) from the HTML library used for unespacing strings
from matplotlib import pyplot as plt # Matplotlib from PyPlot used for certain plotting graphs
from collections import Counter, defaultdict # Counter from collections used to keep counter of # of times a distinct element occurs within a data 
# structure, and defaultdict from collections that is equivalent to the regular dictionary data structure except for being initialized with a 
# particular value for all unknown keys
from nltk.translate import bleu # The NLTK implementation for computing the BLEU score
from nltk.translate.bleu_score import SmoothingFunction # The smoothing function used for BLEU score (is not really used, but only specified in 
# the codes just in case)
from itertools import groupby # groupby(...) from itertools used for splitting a condition based on a certain condition
from symspellpy.symspellpy import SymSpell, Verbosity # The SymSpellPy implementation used for correcting spelling errors in the texts.
# Verbosity from SymSpellPy is only used to make the SymSpellPy output some operational information just in case
import inflect # The Inflect library used to convert digits/numbers into words, e.g. "12" to "twelve"
n2w = inflect.engine() # Initialize an embedding converting digits/numbers into words
import nltk # Import the NLTK library and all its functions
# Load the NLTK corpus for the English language
try: # Try if the NLTK corpus for the English language exists
  nltk.data.find("tokenizers/punkt")
except LookupError: # If the NLTK corpus for the English language was not found
  nltk.download("punkt") # Download the NLTK corpus for the English language

folder_dir = "./" # Current folder
punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'' # All punctuation (as considered by the Keras Tokenizer on default)
allowed_punctuation = ".,:;!?()-" # All punctiation that is considered to be allowed
disallowed_punctuation = punctuation.translate(str.maketrans("", "", allowed_punctuation)) + "´" + "•" + "…" + "“”" + "‘’" + "—" + "❤️❤️❤️" 
# ^ All disallowed punctuation based on partially what was not allowed and based on what other punctuation that was found in the data sets 
REVIEW_START = "<s>" # Start token for a sequence S=<t_1, t_0, ..., t_|S|> where t_1=<s>
REVIEW_END = "</s>" # Termination/end token for a sequence S=<t_1, t_2, ..., t_|S|> where t_|S|=</s>

# Parameters for the SymSpellPy implementation
max_edit_distance_dictionary = 4 # Maximum edit distance
prefix_length = 7 # Prefix length
sym_speller = None # The placeholder for the speller based on the SymSpellPy implementation

"""
The tokenized(...) function returns a list of tokens given a string <s> using the NLTK implementation nltk.word_tokenize(...) function
"""
def tokenized(s):
  return nltk.word_tokenize(s)

"""
The print_data_stats(...) function prints all the relevant stats about the data set <data> with headline <msg>
"""
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
  print("- Standard deviation around number of characters in a drug review:", np.std(lengths), end="\n\n")

"""
The format_text(...) function formats the text <s> using some amount of regex expression that 
e.g. remove repeated number of same punctuation at the same place, extending certain special 
contractions that the PyContractions was not entirely able to do so, and create white spaces 
between punctuation chars that were composite with each other (e.g. ")." to ") ."). Finally, 
the digits/numbers were converted into words using the Inflect library, the final extensions of 
other contractions made using the PyContractions library, and removing disallowed punctuation 
before ultimately returning the formatted text in a tokenized version.
"""
def format_text(s):
  s_new = re.sub("(\s+|^)i\s*(')?\s*d\s+", " i would ", unescape(s[1:-1])) # Special case of contraction (e.g. "i'd"->"i would")
  s_new = re.sub("(\s+|^)i\s*(')?\s*d\s*(')?\s*ve\s+", " i would have ", s_new) # Special case of contraction (e.g. "i'dve"->"i would have")
  s_new = re.sub("(\s+|^)i\s*(')?\s*ll\s+", " i will ", s_new) # Special case of contraction (e.g. "i'll"->"i will")
  s_new = re.sub("(\s+|^)i\s*(')?\s*ll\s*(')?\s*ve\s+", " i will have ", s_new) # Special case of contraction (e.g. "i'llve"->"i will have")
  s_new = re.sub("(\s+|^)i\s*(')?\s*m\s+", " i am ", s_new) # Special case of contraction (e.g. "i'm"->"i am")
  s_new = re.sub("(\s+|^)i\s*(')?\s*ve\s+", " i have ", s_new) # Special case of contraction (e.g. "i've"->"i have")
  s_new = re.sub("\.+", " . ", s_new) # Remove repeated number of "."
  s_new = re.sub("\,+", " , ", s_new) # Remove repeated number of ","
  s_new = re.sub("\?+", " ? ", s_new) # Remove repeated number of "?"
  s_new = re.sub("\!+", " ! ", s_new) # Remove repeated number of "!"
  s_new = re.sub("\:+", " : ", s_new) # Remove repeated number of ":"
  s_new = re.sub("\;+", " ; ", s_new) # Remove repeated number of ";"
  s_new = re.sub("[^\w]*\-+[^\w]*", " - ", s_new) # Create space between "-" and surrounding chars on both sides
  s_new = re.sub("[^\w]*\(+[^\w]*", " ( ", s_new) # Create space between "(" and surrounding chars on both sides
  s_new = re.sub("[^\w]*\)+[^\w]*", " ) ", s_new) # Create space between ")" and surrounding chars on both sides
  s_new = ["".join(g).rstrip(" ") for _, g in groupby(s_new, str.isdigit)] # Split the string after occurrence of numerical digits 
  s_new = " ".join([(n2w.number_to_words(w) if w.isdigit() else w) for w in s_new]).lower() # Convert the digits into words
  s_new = " ".join(tokenized(contractions.fix(s_new))).replace(" '", "") # Fix the final contractions using the PyContractions library
  return " ".join(tokenized(s_new.translate(str.maketrans(disallowed_punctuation, " " * len(disallowed_punctuation))))) # Return tokenized formatted string

"""
This is_valid(...) function is used to determine whether the considered sample <sample> is valid or not for being extracted from a 
certain data set. 
"""
def is_valid(sample):
  not_erronous = not "</span>" in sample[1] if isinstance(sample[1], str) else sample[1]!=-1 # Check the sample
  if not not_erronous: # If the sample is erronous
    return False # Return False indicating that this sample is not valid

  return True # Return True indicating that this sample is valid

"""
The correctly_spelled(...) function returns the corresponding version of the sample (including both the review text to be corrected and 
its associated attributes), whose review text is correctly spelled.
"""
def correctly_spelled(data, max_edit_distance_lookup = None):
  global sym_speller # Make the SymspellPy-based speller global to be able to be used in the body of this function
  if sym_speller is None: # If the speller is not initialized
    sym_speller = SymSpell(max_edit_distance_dictionary, prefix_length) # Initialize the speller provided its parameters as 
    # previously defined
    sym_spell_dict_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt") # Load the frequency dictionary 
    # to the speller
    term_index = 0  # Column of the term in the dictionary text file
    count_index = 1 # Column of the term frequency in the dictionary text file
    if not sym_speller.load_dictionary(sym_spell_dict_path, term_index, count_index): # If the dictionary was not found
      print("ERROR! SymSpellPy dictionary not found at following path:", sym_spell_dict_path) # Print error message informing about this
      os._exit(1) # Exit the entire program

  if max_edit_distance_lookup is None: # If no maximum edit distance during lookup is specified
    max_edit_distance_lookup = max_edit_distance_dictionary # Assign the same edit distance to that as to the maximum edit distance 
    # on the dictionary
  
  # Correct spelling of each token in the text and return the data sample
  return " ".join([(sym_speller.lookup_compound(t, max_edit_distance_lookup)[0].term if t.isalpha() and not(t==data[0] or t==data[1] or ("".join([x[0] for x in data[1].split()])==t if len(data[1].split())>=3 else False)) else t) for t in tokenized(data[2])])

"""
The free_from_infrequents(...) function removes all data samples in data set <data> that occur less than <min_count> times. 
"""
def free_from_infrequents(data, min_count):
  counts = Counter(list(map(tuple, data[:, [0, 1, 3]]))) # Get the counter of each distinct attribute set (on the form (drug, condition, rating))
  new_data = [] # Initialize an empty list containing the new version of the data
  
  for entry in data: # For each sample in <data>
    key = tuple(entry[[0, 1, 3]]) # Identify the attribute set
    if counts[key]>=min_count: # If the attribute set occurs at least <min_count> times
      new_data.append(entry) # Add it to the new_data list
    
  return np.array(new_data) # Return the NumPy version of the new_data list

"""
The get_bleu_score(...) function computes the BLEU score of a drug review <review> against a set of reference drug reviews that are 
selected from the testing data set <test_data> based on having the same attributes as <review>.
"""
def get_bleu_score(review, test_data, verbose = False):
  try: # Try computing the BLEU scores based on available information
    candidate = tokenized(review[2]) # Get the drug review text in <review>
    references = [tokenized(ref_review[2]) for ref_review in test_data if all(ref_review[[0, 1, 3]]==review[[0, 1, 3]])] # Extract all 
    # relevant drug reviews that from <test_data> that have the same attributes as <review>
    return bleu(references, candidate, smoothing_function = SmoothingFunction().method4) # Compute the BLEU score using the NLTK 
    # implementation for computing BLEU scores and return the result 
  except KeyError as e: # If the NLTK implementation for computing BLEU scores could not be run due to no available reference drug reviews
    if verbose:
      print("WARNING: No reference translations were obtainable for the following review:", review) # Print error message
    return -1 # Return -1 to indicate unsuccessful computation

"""
The evaluate_bleu_score(...) function computes the weighted average BLEU score of a fitted model <model> on a testing data set 
<test_data>. The argument <max_review_length> denotes the maximum length of the reviews that the model <model> produces.
"""
def evaluate_bleu_score(model, test_data, max_review_length):
  final_bleu_score = 0 # Initialization of a variable holding the final result
  keys = list(set(map(tuple, test_data[:, [0, 1, 3]]))) # Get the list of distinct attribute sets in <test_data>
  c_keys = Counter(list(map(tuple, test_data[:, [0, 1, 3]]))) # Get the counter of # of times each attribute set occurred in <test_data>

  # Compute the weight that is associated with the attribute set with respect to its frequency within <test_data>
  def weight(key):
    return c_keys[key]/sum(c_keys.values())
  
  for i, key in enumerate(keys): # For each distinct attribute set
    review = np.array([key[0], key[1], " " * max_review_length * 5, key[2]]).reshape(1, -1) # Initialize the ground drug review provided 
    # available information thus far
    generated_review = model.generate(review)[0] # Generate a drug review text using <model>
    bleu_score = get_bleu_score(generated_review, test_data) # Calculate the BLEU score for the newly generated drug review text using 
    # <test_data>
    final_bleu_score += bleu_score * weight(key) # Add the weighted BLEU score to the variable holding the final result
    if i%5==0: # For each 5th iteration
      print(generated_review) # Print a sample drug review text generated by <model>
  
  return final_bleu_score # Return the final result