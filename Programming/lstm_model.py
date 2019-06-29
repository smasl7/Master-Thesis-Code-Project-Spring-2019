"""
Author: Sylwester Liljegren
Date: 2019-06-29

This is the implementation of the LSTM model that was used in the experiments.
The LSTM model for the experiments was implemented as a word-level Long Short-Term 
Memory network that made use of GloVe word embeddings as it was observed to enhance the generative 
performance of the model and to also speed up the calculations during training/testing.
The implementation was partially based on the tutorial provided by Shivam Bansal from Medium.
The link to the tutorial could be found below:
https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

As mentioned, it used GloVe worde embeddings. The source of these word embeddings could be found in the following 
link: https://nlp.stanford.edu/projects/glove/
"""

### LIBRARIES ###
from keras.layers import Embedding, LSTM, Dense, Dropout # The relevant layers from the Keras library
from keras.preprocessing.sequence import pad_sequences # pad_sequences(...) from the Keras library pads 
# sequences of variable length to a sequence of a fixed length
from keras.preprocessing.text import Tokenizer # Tokenizer from the Keras library that manages texts that 
# are fed to the Keras tokenizer
from keras.callbacks import EarlyStopping # Early stopping from the Keras library used for stopping training 
# when the validation error has ceased to decrease
from keras.models import Sequential # A Keras API for simpler models
from collections import Counter # Counter from collections used to keep counter of # of times a distinct element occurs within a data 
# structure
from utils import * # Import all functions and values from utils.py
from copy import deepcopy # Copy library
import keras.utils as ku # Some fundamental functions that are defined by the Keras library
import numpy as np # NumPy library
import os

glove_dir = folder_dir + "glove.6B" # The path to the GloVe word embeddings

class LSTM_model:
  """
  The __init__(...) function initializes any instance of the LSTM_model class with certain arguments that the user 
  is free to define however it wants.
  """
  def __init__(self, num_units = 256, drop_out = 0.1, embedding_size = 50, embedding_tune = False):
    ### PREPROCESSING ###
    self.tokenizer = Tokenizer(filters = disallowed_punctuation) # The Keras tokenizer responsible for managing textual data
    self.num_tokens = None # Number of unique tokens
    self.max_sequence_len = None # Maximum length of a sequence
    
    ### MODEL ###
    self.num_units = num_units # Number of units in the LSTM model
    self.drop_out = drop_out # The dropout rate within the model
    self.embedding_size = embedding_size # Size of the word embeddings
    self.embedding_tune = embedding_tune # Bool value indicating whether the word embeddings should be trained or not
    self.model = None # The model itself that has been implemented in Keras and used for generating drug review texts
    self.history = None # The training history of the model itself
  
  """
  The __preprocess(...) function preprocess the texts that are present in the training data set <train_data> and returns 
  a corresponding preprocessed training data set that is able to be processed by the LSTM model afterwards.
  """
  def __preprocess(self, train_data, verbose = True):
    train_data = deepcopy(REVIEW_START + " " + train_data + " " + REVIEW_END) # Include the start- and end tokens
    self.tokenizer.fit_on_texts(train_data) # Fit the Keras tokenizer on the training data set
    self.num_tokens = len(self.tokenizer.word_index) + 1 # Obtain the number of tokens in the training data set
    if verbose: # If verbosity is activated
      print("- Number of tokens:", self.num_tokens)
    
    sequenced_train_data = np.array(self.tokenizer.texts_to_sequences(train_data)) # Convert the texts to sequences
    self.max_sequence_len = max(len(review) for review in sequenced_train_data) # Get the greatest length among the sequences

    input_sequences = [] # Initialize an empty list holding the input sequences
    for review in sequenced_train_data: # For each drug review (as a sequence)
      for t in range(1, len(review)): # For each t in [1..len(review)-1]
        input_sequences.append(review[:t+1]) # Append the input_sequences list with a sliced sequence review[:t+1]
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre')) # Zero pad all the sequences to 
    # have the same length as the longest sequence
    
    predictors, labels = input_sequences[:,:-1], input_sequences[:,-1] # Split the input_sequences into predictors and labels
    labels = ku.to_categorical(labels, num_classes=self.num_tokens) # Convert the labels to probability vectors
    return predictors, labels # Return predictors and labels

  """
  The __glove_embedding_matrix(...) function returns an embeddings matrix that for each token contains its own GloVe word embedding.
  """
  def __glove_embedding_matrix(self, verbose):
    embeddings_index = {} # Initialize the dictionary for holding the embedding indices
    
    # Read from the file that contains the GloVe word embeddings having a certain dimensionality size
    with open(os.path.join(glove_dir, "glove.6B." + str(self.embedding_size) + "d.txt"), encoding="utf8") as f:
      for line in f: # For each line in the text file
        values = line.split() # Split the read line with respect to white spaces
        word = values[0] # Get the word from the line
        coefs = np.asarray(values[1:], dtype="float32") # Get the actual GloVe word embedding
        embeddings_index[word] = coefs # Include the mapping between the word and the GloVe word embedding using the dictionary 
        # embeddings_index
    
    embedding_matrix = np.random.rand(self.num_tokens, self.embedding_size) # Initialize a matrix containing values that are uniformly 
    # distributed on [0,1]
    non_existent_sum = 0 # Number of words, for whom word embeddings could not be found
    for word, i in self.tokenizer.word_index.items(): # For each word and its index as hold by the tokenizer
      embedding_vector = embeddings_index.get(word) # Get the word embedding vector for the word
      if not embedding_vector is None: # If some word embedding could be found for the word
        embedding_matrix[i] = embedding_vector # Fill out the chosen row in the matrix with the word embedding
      elif verbose: # If verbosity is activated
        print("NOTE! Embedding could not be found for following word:", word) # Inform about the fact that no word embedding could be 
        # found for the word
        non_existent_sum += 1 # Increment the counter over # of words, for whom word embeddings could not be found
    
    if verbose: # If verbosity is activated
      print("- Number of words for which embeddings could not be found:", non_existent_sum)
    
    return embedding_matrix # Return the word embedding matrix

  """
  The fit(...) function trains the model provided a training data set <train_data> that is given as argument for some number of epochs 
  <num_epochs>.
  """
  def fit(self, train_data, num_epochs = 50, verbose = True):
    if verbose: # If verbosity is activated
      print("Preprocessing the data for the LSTM model...")
    
    # Preparation of the data
    predictors, labels = self.__preprocess(train_data, verbose) # Return a preprocessed version of the texts
    glove_embed_matrix = self.__glove_embedding_matrix(verbose) # Return the word embedding matrix
    
    if verbose: # If verbosity is activated
      print("Done with preparing the data for the LSTM model!", end="\n\n")
    
    # Definition and training of the model
    if verbose:
      print("Training the LSTM model...")

    self.model = Sequential() # Initialization of the model itself
    self.model.add(Embedding(self.num_tokens, self.embedding_size, weights = [glove_embed_matrix], input_length = self.max_sequence_len - 1, mask_zero = True, trainable = self.embedding_tune)) # Embedding layer
    self.model.add(LSTM(self.num_units)) # LSTM layer
    self.model.add(Dropout(self.drop_out)) # Dropout layer
    self.model.add(Dense(self.num_tokens, activation='softmax')) # Output layer
    self.model.compile(loss='categorical_crossentropy', optimizer="adam")
    self.history = self.model.fit(predictors, labels, epochs = num_epochs, callbacks = [
      EarlyStopping(patience = 2, restore_best_weights = True, verbose = 1 if verbose else 0)
    ], validation_data = (predictors, labels), shuffle = True, verbose = 1 if verbose else 0) # Traind and assign the history of training the model to self.history

  """
  The generate(...) function generates drug reviews provided the input attributes in <reviews> assuming that the model has been 
  previously trained.
  """
  def generate(self, reviews, random_predict = False):
    predicted_reviews = deepcopy(reviews) # Make a deep copy of the input drug reviews to another variable to risk not being 
    # modified by reference

    # A function that performs a prediction with the trained model given a preprocessed sequence
    def predict_token(token_list, random):
      ps = self.model.predict(token_list, verbose = 0)[0] # Make a prediction
      return np.random.choice([i for i in range(self.num_tokens)], p = ps) if random else np.argmax(ps) # If non-random, 
      # then select the token that maximizes the probability of appearing next in the sequence. Otherwise, if random, 
      # then randomly select a token with respect to the probability of each token to appear next in the sequence

    # This function converts a token to a word
    def from_token_to_word(token):
      output_word = "" # Initialization of the variable output_word
      for word, index in self.tokenizer.word_index.items(): # For each word and its index as hold by the tokenizer
        if index == token: # If the token and the index is the same
          output_word = word # Assign the word to output_word
          break # Exit the loop
      
      return output_word # Return the word
    
    for i in range(reviews.shape[0]): # For each input drug review
      generated = REVIEW_START # Initialize the sequence to be built having the start token ("<s>") as the first token
      for j in range(self.max_sequence_len-1): # For each step in the sequence to be built
        token_list = self.tokenizer.texts_to_sequences([generated])[0] # Convert the current text to a sequence of tokens
        token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre') # Pad the sequence
        next_token = predict_token(token_list, j==0 or random_predict) # Predict token
        next_word = from_token_to_word(next_token) # Get the word out of the predicted token
        if next_word==REVIEW_END: # If the end token was predicted
          break # Exit the loop
        else: # If something else was predicted
          generated += " " + next_word # Add the next word to the current text
      predicted_reviews[i,2] = generated.lstrip(REVIEW_START).lstrip() # Assign the random text to correct positions in the concerned input drug reviews
    
    return predicted_reviews # Return the completed drug reviews