"""
Author: Sylwester Liljegren
Date: 2019-07-01

This is the implementation of the V-Att2Seq model that was used in the experiments. The implementation of the 
V-Att2Seq model as seen in this file is based on the description of its architecture and involving mechanisms 
as given in the master thesis report that is attached along with this code project. Each drug review text is 
interpreted as a sequence of tokens, where tokens are either punctuation or single words. For further information, 
please check the master thesis report describing the design of the V-Att2Seq model in depth. Each token also 
used GloVe worde embeddings. The source of these word embeddings could be found in the following 
link: https://nlp.stanford.edu/projects/glove/

For implementing the variational recurrent autoencoder, the following implementation of corresponding 
architecture found at the following link: https://github.com/twairball/keras_lstm_vae/blob/master/lstm_vae/vae.py 
was used. Some minor modifications had to be performed to successfully integrate that implementation with the overall
implementation of the V-Att2Seq model.
"""

### LIBRARIES ###
from keras.layers import Input, LSTM, Lambda, RepeatVector, Dense, Concatenate, Dropout # The relevant layers from the Keras library
from keras.preprocessing.sequence import pad_sequences # pad_sequences(...) from the Keras library pads 
# sequences of variable length to a sequence of a fixed length
from keras.preprocessing.text import Tokenizer # Tokenizer from the Keras library that manages texts that 
# are fed to the Keras tokenizer
from keras.callbacks import EarlyStopping # Early stopping from the Keras library used for stopping training 
# when the validation error has ceased to decrease
from keras.models import Model # A Keras API for simpler models
from keras import objectives
from utils import * # Import all functions and values from utils.py
from copy import deepcopy # Copy library
from sklearn.preprocessing import OneHotEncoder
import keras.utils as ku # Some fundamental functions that are defined by the Keras library
import keras.backend as K
import numpy as np # NumPy library
import os

glove_dir = folder_dir + "glove.6B" # The path to the GloVe word embeddings

class V_Att2Seq_model:
  """
  The __init__(...) function initializes any instance of the V-Att2Seq_model class with certain arguments that the user 
  is free to define however it wants.
  """
  def __init__(self, embedding_dim = 50, vrae_intermediate_dim = 256, vrae_hidden_dim = 256, main_encoding_dim = 64, main_attribute_dim = 192, main_hidden_dim = 350, main_dropout = 0.1):
    # PREPROCESSING
    self.tokenizer = Tokenizer(filters = disallowed_punctuation)
    self.embedding_dim = embedding_dim # Dimensionality of the GloVe word embeddings to use
    self.embedding_matrix = None # Glove word embedding matrix
    self.num_tokens = None # Number of tokens detected in the training data set
    self.max_sequence_len = None # Maximum length of all sequences in the training data set
    self.onehot_drug_encoder = None # One-hot encoder of the drug attribute
    self.onehot_cond_encoder = None # One-hot encoder of the condition attribute
    self.onehot_rating_encoder = None # One-hot encoder of the rating attribute

    # VARIATIONAL RECURRENT AUTOENCODER
    self.vrae_intermediate_dim = vrae_intermediate_dim # Dimensionality of the intermediate layer between input/output layer and the hidden layer
    self.vrae_hidden_dim = vrae_hidden_dim # Dimensionality of the hidden layer
    self.vrae_encoder = None # Encoder
    self.vrae_history = None # Training progress history of the variational recurrent autoencoder

    # MAIN MODEL
    self.main_encoding_dim = main_encoding_dim # Dimensionality of the encoding of the attributes
    self.main_attribute_dim = main_attribute_dim # Dimensionality of the attribute vector composing the encoded attributes
    self.main_hidden_dim = main_hidden_dim # Dimensionality of the main hidden vector within the V-Att2Seq model
    self.main_dropout = main_dropout # Dropout rate on the second-last layer before the output layer
    self.main_model = None # Main model
    self.main_history = None # Training progress history of the main model
  
  """
  The __preprocess(...) function preprocess the texts and attributes that are present in the training data set <train_data> and returns 
  a corresponding preprocessed training data set that is able to be processed by the V-Att2Seq model afterwards.
  """
  def __preprocess(self, train_data, verbose = True):
    train_data[:,2] = deepcopy(REVIEW_START + " " + train_data[:,2] + " " + REVIEW_END) # Include the start- and end tokens
    if verbose: # If verbosity is activated
        print("- Number of tokens:", self.num_tokens)
    
    sequenced_train_data_texts = np.array(self.tokenizer.texts_to_sequences(train_data[:,2])) # Convert the texts to sequences
    self.max_sequence_len = max(len(review) for review in sequenced_train_data_texts) # Get the greatest length among the sequences

    drugs = [] # Initialize the list for the drug attributes
    conds = [] # Initialize the list for the condition attributes
    ratings = [] # Initialize the list for the rating attributes
    input_sequences = [] # Initialize the list for the sequences
    for i, review in enumerate(sequenced_train_data_texts): # For each sequenced drug review
        for t in range(1, len(review)): # For each t in [1..len(review)-1]
            drugs.append(train_data[i,0]) # Append the drug list with the current drug attribute
            conds.append(train_data[i,1]) # Append the condition list with the current condition attribute
            ratings.append(train_data[i,3]) # Append the rating list with the current rating attribute
            input_sequences.append(review[:t+1]) # Append the sequence list with the current slice of the sequence
    input_sequences = np.array(pad_sequences(input_sequences, maxlen = self.max_sequence_len, padding="pre")) # Pad the sequences

    predictors, labels = input_sequences[:,:-1], input_sequences[:,-1] # Split the input sequences into predictors and labels
    categorical_labels = ku.to_categorical(labels, num_classes = self.num_tokens) # Convert the token identifiers into categorical vectors
    embedded_predictors = np.zeros((predictors.shape[0], self.max_sequence_len-1, self.embedding_dim)) # Initialize the embedded predictors matrix
    for i, seq in enumerate(predictors): # For each predictor sequence
        for j, t in enumerate(seq): # For each token in the predictor sequence
            embedded_predictors[i,j] = self.embedding_matrix[t] # Assign the GloVe word embedding of the token at time step j in the predictor sequence
    
    # Return the results
    return np.array(drugs).reshape((-1, 1)), np.array(conds).reshape((-1, 1)), np.array(ratings).reshape((-1, 1)), embedded_predictors, categorical_labels

  """
  The __glove_embedding_matrix(...) function returns an embeddings matrix that for each token contains its own GloVe word embedding.
  """
  def __glove_embedding_matrix(self, verbose):
    embeddings_index = {} # Initialize the dictionary for holding the embedding indices
    
    # Read from the file that contains the GloVe word embeddings having a certain dimensionality
    with open(os.path.join(glove_dir, "glove.6B." + str(self.embedding_dim) + "d.txt"), encoding="utf8") as f:
      for line in f: # For each line in the text file
        values = line.split() # Split the read line with respect to white spaces
        word = values[0] # Get the word from the line
        coefs = np.asarray(values[1:], dtype="float32") # Get the actual GloVe word embedding
        embeddings_index[word] = coefs # Include the mapping between the word and the GloVe word embedding using the dictionary 
        # embeddings_index
    
    embedding_matrix = np.random.rand(self.num_tokens, self.embedding_dim) # Initialize a matrix containing values that are uniformly 
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
  def fit(self, train_data, vrae_num_epochs = 50, main_num_epochs = 50, verbose = True):
    self.tokenizer.fit_on_texts(train_data[:,2]) # Fit the Keras tokenizer on the training data set
    self.num_tokens = len(self.tokenizer.word_index) + 1 # Obtain the number of tokens in the training data set
    self.embedding_matrix = self.__glove_embedding_matrix(verbose) # Obtain the GloVe word embedding matrix
    drugs, conds, ratings, seqs, next_tokens = self.__preprocess(train_data, verbose = 1 if verbose else 0) # Obtain preprocessed data set

    self.onehot_drug_encoder = OneHotEncoder(sparse = False) # Initialize the one-hot encoder for the drug attribute
    onehotted_drugs = self.onehot_drug_encoder.fit_transform(drugs) # Learn and produce one-hot encodings for the drug attributes
    self.onehot_cond_encoder = OneHotEncoder(sparse = False) # Initialize the one-hot encoder for the condition attribute
    onehotted_conds = self.onehot_cond_encoder.fit_transform(conds) # Learn and produce one-hot encodings for the condition attributes
    self.onehot_rating_encoder = OneHotEncoder(sparse = False) # Initialize the one-hot encoder for the rating attribute
    onehotted_ratings = self.onehot_rating_encoder.fit_transform(ratings) # Learn and procue one-hot encodings for the rating attributes

    self.__fit_vrae(seqs, vrae_num_epochs, verbose = 1 if verbose else 0) # Train the variational recurrent autoencoder to get the 
    # encoder to be used in the subsequent procedures

    """
    MAIN MODEL AS DEFINED USING THE KERAS LIBRARY
    """
    input_drug_attr = Input(shape = (onehotted_drugs.shape[1],)) # Input layer for the drug attributes (one-hot encoded)
    encoded_input_drug_attr = Dense(self.main_encoding_dim, use_bias = False)(input_drug_attr) # Encoded drug attribute

    input_cond_attr = Input(shape = (onehotted_conds.shape[1],)) # Input layer for the condition attributes (one-hot encoded)
    encoded_input_cond_attr = Dense(self.main_encoding_dim, use_bias = False)(input_cond_attr) # Encoded condition attribute

    input_rating_attr = Input(shape = (onehotted_ratings.shape[1],)) # Input layer for the rating attributes (one-hot encoded)
    encoded_input_rating_attr = Dense(self.main_encoding_dim, use_bias = False)(input_rating_attr) # Encoded rating attribute

    combined_encoded_attrs = Concatenate()([encoded_input_drug_attr, encoded_input_cond_attr, encoded_input_rating_attr]) # Concatenation of encoded attributes
    main_attr_vec = Dense(self.main_attribute_dim, activation = "tanh")(combined_encoded_attrs) # Main attribute vector

    input_sequence = Input(shape = (self.max_sequence_len-1, self.embedding_dim,)) # Input layer for the current sequence
    vrae_hidden_vec = self.vrae_encoder(input_sequence) # Hidden vector as a result of encoding the current sequence

    combined_vecs = Concatenate()([main_attr_vec, vrae_hidden_vec]) # Concatenation of the hidden vector from the sequence encoder and the main attribute vector
    main_hidden = Dense(self.main_hidden_dim, activation = "tanh")(combined_vecs) # Main hidden vector
    prob_dist = Dense(self.num_tokens, activation = "softmax", use_bias = False)(main_hidden) # Softmax vectors using the main hidden vectors

    self.main_model = Model([input_drug_attr, input_cond_attr, input_rating_attr, input_sequence], prob_dist) # The functional main model
    self.main_model.compile(optimizer="adam", loss="categorical_crossentropy") # Compiling the main model
    # Training the model and storing its training progress history to self.main_history
    self.main_history = self.main_model.fit([onehotted_drugs, onehotted_conds, onehotted_ratings, seqs], next_tokens, epochs = main_num_epochs, 
    callbacks = [
        EarlyStopping(patience = 2, restore_best_weights = True, verbose = 1 if verbose else 0)
    ], validation_data = ([onehotted_drugs, onehotted_conds, onehotted_ratings, seqs], next_tokens), verbose = 1 if verbose else 0)

    # Return the training progress history of the main model
    return self.main_history

  def __fit_vrae(self, train_data, num_epochs, verbose):
    x = Input(shape=(self.max_sequence_len-1, self.embedding_dim,)) # Input layer for sequences

    h_in = LSTM(self.vrae_intermediate_dim)(x) # LSTM encoding of the sequences
    z_mean = Dense(self.vrae_hidden_dim)(h_in) # Mean vector given the LSTM encoding
    z_log_sigma = Dense(self.vrae_hidden_dim)(h_in) # Log-sigma vector given the LSTM encoding

    # This functions samples a hidden vector from a continous space using both the mean vector and the log-sigma vector
    def sampling(args):
        z_mean, z_log_sigma = args # Parse the arguments
        epsilon = K.random_normal(shape=(self.vrae_hidden_dim,), mean=0, stddev=1) # Obtain a standard-normally distributed value
        return z_mean + z_log_sigma * epsilon # Return a sampled hidden vector

    z = Lambda(sampling, output_shape=(self.vrae_hidden_dim,))([z_mean, z_log_sigma]) # Sample a hidden vector

    z_repeated = RepeatVector(self.max_sequence_len-1)(z)
    h_out = LSTM(self.vrae_intermediate_dim, return_sequences=True)(z_repeated)
    x_decoded_mean = LSTM(self.embedding_dim, return_sequences=True)(h_out)

    vrae = Model(x, x_decoded_mean) # End-to-end variational recurrent autoencoder
    self.vrae_encoder = Model(x, z) # Encoder part of the variational recurrent autoencoder (i.e. from inputs to hidden space)

    # This function calculates the VRAE loss between the input and the reconstructed input by calculating Mean Squared Error (MSE) 
    # between them and calculating the Kullback-Leibler loss between the inputs, which both of them are added altogether to get 
    # the final loss value
    def vrae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean) # Calculate the MSE
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)) # Calculate the Kullback-Leibler loss
        loss = xent_loss + kl_loss # Sum the two calculated results together
        return loss # Return the final loss

    vrae.compile(optimizer='rmsprop', loss = vrae_loss, metrics = [vrae_loss]) # Compiling the variational recurrent autoencoder
    # Train the model and store the training progress history into self.vrae_history variable
    self.vrae_history = vrae.fit(train_data, train_data, epochs = num_epochs, validation_split = 0.1, callbacks = [
        EarlyStopping(monitor="val_vrae_loss", patience = 2, restore_best_weights = True, verbose = 1 if verbose else 0)
    ], verbose = 1 if verbose else 0)

    # Transfer all the encoder weights from the end-to-end variational recurrent autoencoder into the variable self.vrae_encoder
    self.vrae_encoder.layers[0].set_weights(vrae.layers[0].get_weights())
    self.vrae_encoder.layers[1].set_weights(vrae.layers[1].get_weights())
    self.vrae_encoder.layers[2].set_weights(vrae.layers[2].get_weights())
    self.vrae_encoder.layers[3].set_weights(vrae.layers[3].get_weights())
    self.vrae_encoder.layers[4].set_weights(vrae.layers[4].get_weights())

    return self.vrae_history
  
  """
  The generate(...) function generates drug reviews provided the input attributes in <reviews> assuming that the model has been 
  previously trained.
  """
  def generate(self, reviews, random_predict = False):
    predicted_reviews = deepcopy(reviews) # Make a deep copy of the input drug reviews to another variable to risk not being 
    # modified by reference

    # A function that performs a prediction with the trained model given a preprocessed sequence
    def predict_token(drug, cond, rating, token_list, random):
      ps = self.main_model.predict([drug, cond, rating, token_list], verbose = 0)[0] # Make a prediction
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
      onehot_drug = self.onehot_drug_encoder.transform(reviews[i,0].reshape((-1,1)))
      onehot_cond = self.onehot_cond_encoder.transform(reviews[i,1].reshape((-1,1)))
      onehot_rating = self.onehot_rating_encoder.transform(reviews[i,3].reshape((-1,1)))

      for j in range(self.max_sequence_len-1): # For each step in the sequence to be built
        token_list = self.tokenizer.texts_to_sequences([generated])[0] # Convert the current text to a sequence of tokens
        token_list = np.array(pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')) # Pad the sequence
        embedded_token_list = np.zeros((1, self.max_sequence_len-1, self.embedding_dim))
        for j, t in enumerate(token_list[0]):
            embedded_token_list[0,j] = self.embedding_matrix[t]

        next_token = predict_token(onehot_drug, onehot_cond, onehot_rating, embedded_token_list, random_predict) # Predict token
        next_word = from_token_to_word(next_token) # Get the word out of the predicted token
        if next_word==REVIEW_END: # If the end token was predicted
          break # Exit the loop
        else: # If something else was predicted
          generated += " " + next_word # Add the next word to the current text
      predicted_reviews[i,2] = generated.lstrip(REVIEW_START).lstrip() # Assign the random text to correct positions in the concerned input drug reviews
    
    return predicted_reviews # Return the completed drug reviews
