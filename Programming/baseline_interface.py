from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import RMSprop
import keras.utils as ku
import numpy as np

folder_dir = "./"
baseline_dir = folder_dir + "baseline/"

##### Long Short-Term Memory network: The baseline model - Preparation: Training/optimizing of the Long Short-Term Memory network #####
def is_valid_for_baseline(sample, training = True):
  if training:
    return True
  else:
    return True

class LSTM_model:
  def __init__(self, word_level = False, embedding_size = None, num_units = 250, drop_out = 0.1):
    self.word_level = word_level
    self.tokenizer = Tokenizer(char_level = not self.word_level)
    self.num_tokens = None
    self.max_sequence_len = None
    
    self.embedding_size = embedding_size
    self.num_units = num_units
    self.drop_out = drop_out
    self.model = None
  
  def __preprocess(self, train_data, verbose = True, predicting = False):
    if verbose:
      print("Preprocessing the data for the LSTM model...")

    self.tokenizer.fit_on_texts(train_data)
    self.num_tokens = len(self.tokenizer.word_index) + 1
    
    if verbose:
      print(self.num_tokens)
    
    sequenced_train_data = np.array(self.tokenizer.texts_to_sequences(train_data))
    self.max_sequence_len = max(len(review) for review in sequenced_train_data)

    if self.embedding_size is None:
      input_sequences = np.zeros((sequenced_train_data.shape[0], self.max_sequence_len, self.num_tokens), dtype=np.bool)
      
      for i, review in enumerate(sequenced_train_data):
        for t in range(len(review)):
          input_sequences[i, t, review[t]] = 1
      
      if predicting:
        return input_sequences
      else:
        return input_sequences[:, :-1, :], input_sequences[:, -1, :]
    else:
      input_sequences = []
      for review in sequenced_train_data:
        for t in range(1, len(review)):
          input_sequences.append(review[:t+1])
      
      input_sequences = np.array(pad_sequences(input_sequences, maxlen=max(len(x) for x in input_sequences), padding='pre'))
      if predicting:
        return input_sequences
      else:
        predictors, labels = input_sequences[:,:-1], input_sequences[:,-1]
        labels = ku.to_categorical(labels, num_classes=self.num_tokens)
        return predictors, labels

  def fit(self, train_data, num_epochs = 1, verbose = True, save = False):
    # Preparation of the data
    predictors, label = self.__preprocess(train_data, verbose)
    
    if verbose:
      print("Done with preparing the data for the LSTM model!", end="\n\n")
    
    # Definition and training of the model
    if verbose:
      print("Training the LSTM model...")

    self.model = Sequential()
    if not self.embedding_size is None:
      self.model.add(Embedding(self.num_tokens, self.embedding_size, input_length = self.max_sequence_len - 1))
      self.model.add(LSTM(self.num_units))
    else:
      self.model.add(LSTM(self.num_units, input_shape = (self.max_sequence_len - 1, self.num_tokens)))
    self.model.add(Dropout(self.drop_out))
    self.model.add(Dense(self.num_tokens, activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
    self.model.fit(predictors, label, epochs = num_epochs, verbose = 1 if verbose else 0)
    
    if verbose:
      print("Done with training the LSTM model!", end="\n\n")
    
    if save:
      if verbose:
        print("Saving the model...")

      model_path = baseline_dir + "LSTM_" + ("word" if self.word_level else "char") + "_embedd-" + str(self.embedding_size) + "_units-" + str(self.num_units) + "_dropo-" + str(self.drop_out) + ".h5"
      self.model.save(model_path)

      if verbose and save:
        print("Model saved! It has the following path:", model_path)
  
  def generate(self, seed, num_next_tokens, temperature):
    def generate_token(preds):
      # helper function to sample an index from a probability array
      preds = np.asarray(preds).astype('float64')
      preds = np.log(preds) / temperature
      exp_preds = np.exp(preds)
      preds = exp_preds / np.sum(exp_preds)
      probas = np.random.multinomial(1, preds, 1)
      return np.argmax(probas)
    
    generated = seed
    for _ in range(num_next_tokens):
      y = self.model.predict_proba(self.__preprocess([generated])[0], False, True)
      next_token = self.tokenizer.sequences_to_texts([generate_token(y)])[0]
      generated += (" " if self.word_level else "") + next_token

    return generated