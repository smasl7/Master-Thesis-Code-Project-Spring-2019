"""
Author: Sylwester Liljegren
Date: 2019-06-29

This is the implementation of the NN model that was used in the experiments in the main file.
"""

### LIBRARIES ###
import numpy as np # NumPy library
from copy import deepcopy # Copy library

class NN_model:
    """
    The __init__(self) function initializes any instance of the NN_model class with no arguments.
    """
    def __init__(self):
        self.train_data = None # Initialize the placeholder for training data
    
    """
    The fit(...) function trains the model using the training data set <train_data>
    """
    def fit(self, train_data):
        self.train_data = deepcopy(train_data) # Just store the training data set in the variable <self.train_data>
    
    """
    The __partitioned_text_data(self, review) function extracts a partition of the stored training data set where the set of 
    drug reviews have the same attributes as the input drug review <review>
    """
    def __partitioned_text_data(self, review):
        texts = [] # Initialize an empty list holding the texts to return
        for entry in self.train_data: # For each sample in <self.train_data>
            if all(entry[[0, 1]]==review[[0, 1]]) and float(review[3])==float(entry[3]): # If the attributes in the sample and 
                # the input drug reivew are the same
                texts.append(entry[2]) # Append the list with the sample
        
        return texts # Return the list with all appended samples
    
    """
    The generate(...) function generates drug reviews provided the input attributes in <reviews> assuming that the model has been 
    previously trained.
    """
    def generate(self, reviews):
        predicted_reviews = deepcopy(reviews) # Make a deep copy of the input drug reviews over to a separate variable to risk not 
        # become modified by reference
        for i in range(reviews.shape[0]): # For each input drug review
            texts = self.__partitioned_text_data(reviews[i]) # Extract all the drug reviews in the training data set that have the 
            # same attributes as the input drug reviews
            random_text = np.random.choice(texts) # Select a random drug review from that partition
            predicted_reviews[i,2] = random_text # Assign the random text to correct positions in the concerned input drug reviews

        return predicted_reviews # Return the completed drug reviews