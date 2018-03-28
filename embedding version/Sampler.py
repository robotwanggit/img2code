from __future__ import print_function
from __future__ import absolute_import

from Dataset import Dataset
from Utils import *
from Config import *
from Vocabulary import Vocabulary
import numpy as np

class Sampler(object):
  #Retrieve the vocabulary built during training, and initialize the input_shape, output_size and context_length of the model.
  def __init__(self, voc_path, input_shape, output_size, context_length):
    self.voc = Vocabulary()
    self.voc.retrieve(voc_path)
    self.input_shape = input_shape
    self.output_size = output_size
    
    print("Vocabulary size:{}".format(self.voc.size))
    print("Input shape：{}".format(self.input_shape))
    print("Output size: {}".format(self.output_size))
    
    self.context_length = context_length
  
  #predict the token sequences corresponding to the input_img based on the provided model, 
  #the token sequence generation stops if END_TOKEN encountered
  #or sequence_length tokens have been generated
  def predict_greedy(self, model, input_img, require_sparse_label = False, sequence_length = 150, verbose = False):
    current_context = [self.voc.vocabulary[PLACEHOLDER]]* (self.context_length -1) #stores the current token segments in format index or one-hot encoding
    current_context.append(self.voc.vocabulary[START_TOKEN])
    if require_sparse_label:
      current_context = one_hot(current_context, self.output_size)
    #print(np.array([current_context]))
    predictions = START_TOKEN   #stores the token sequence in string format
    out_probs = []
    for i in range(0, sequence_length):
      if verbose:
        print("predicting {}/{}...".format(i, sequence_length))
      probs = model.predict(input_img, np.array([current_context]))
      prediction = np.argmax(probs)
      out_probs.append(probs)
      
      new_context = []
      new_context += current_context[1:self.context_length]
      new_context += (one_hot([prediction], self.output_size) if require_sparse_label else [prediction])
      
      current_context = new_context
      token =  self.voc.token_lookup[prediction]
      predictions += (" " + token)
      
      if token == END_TOKEN:
        break
    return predictions, out_probs
    
      