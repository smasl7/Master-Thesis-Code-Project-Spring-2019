from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.utils as ku
import numpy as np

folder_dir = "./"
sugg_dir = folder_dir + "suggested alternative/"
import sys
sys.path.insert(0, folder_dir + "keras_lstm_vae-master")
from lstm_vae.vae import create_lstm_vae

def is_valid_for_suggestion(sample, training = True):
  if training:
    return True
  else:
    return True

class suggested_alternative:
  def __init__(self, word_level = False):
    self.word_level = word_level
    self.tokenizer = Tokenizer(char_level = not self.word_level)
    self.max_sequence_len = None
    
    # Settings for the recurrent variational autoencoder
    self.encoder = None
    self.decoder = None
    
    # Settings for the collaborative filtering algorithm
    self.cf = None
    
  def __preprocess(self, train_data, verbose = True):
    if verbose:
      print("Preprocessing the data for the LSTM model...")
    
    self.tokenizer.fit_on_texts(train_data)
    num_tokens = len(self.tokenizer.word_index) + 1
    
    if verbose:
      print(num_tokens)
    
    input_sequences = []
    for review in train_data:
      input_sequence = []
      token_list = self.tokenizer.texts_to_sequences([review])[0]
      for i in range(1, len(token_list)):
          n_gram_sequence = token_list[:i+1]
          input_sequence.append(n_gram_sequence)
      input_sequences.append(input_sequence)
    
    if verbose:
      print(len(input_sequences))
    
    self.max_sequence_len = max(len(input_seq) for input_seq in input_sequences)
    input_sequences = np.array(pad_sequences([pad_sequences(input_seq,
                                             maxlen = self.max_sequence_len,
                                             padding='pre') for input_seq in input_sequences], 
                                             maxlen = self.max_sequence_len,
                                             padding='pre'))
    
    return input_sequences
  
  def fit(self, train_data, verbose = True):
    # One-hot encoding of both the drugs and conditions associated with the reviews
    """onehot_drugs = OneHotEncoder().fit_transform(train_data[:,0].reshape(-1, 1))
    onehot_conds = OneHotEncoder().fit_transform(train_data[:,1].reshape(-1, 1))"""
    
    # Transforming the review texts into latent numericals vectors
    self.__fit_lstm(self.__preprocess(train_data[:,2]))
    self.encoder = Model()
  
  def __fit_lstm(self, train_data):
    print(train_data.shape)
    lstm_vae, _, _ = create_lstm_vae(train_data.shape[-1], train_data.shape[1], 32, 250, 1)
    lstm_vae.summary()
    lstm_vae.fit(train_data)
  
  def __fit_cf(self, train_data):
    return train_data
    
  def generate(self, x):
    return x