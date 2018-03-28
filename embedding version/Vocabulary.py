import numpy as np
import sys
from Config import *

SEPARATOR = "->"        #symbol to separate the key and value when storing the vocabulary dict in a file

class Vocabulary(object):

  #Initialize members in a Vocabulary class
  #binary_vocabulary is a dict which maps token string to index in one-hot format
  #vocabulary is a dict which maps token string to index
  #token_lookup is a dict which maps index to token string
  def __init__(self):
    self.binary_vocabulary = {}
    self.vocabulary = {}
    self.token_lookup = {}
    self.size = 0
    
    self.append(START_TOKEN)
    self.append(END_TOKEN)
    self.append(PLACEHOLDER)
    
  def append(self, token):
    if token not in self.vocabulary:
      self.vocabulary[token] = self.size
      self.token_lookup[self.size] = token
      self.size+=1
  
  #create a vocabulary containing maps from token to its one-hot representation
  def create_binary_representation(self):
    if sys.version_info >= (3,):
      items = self.vocabulary.items()
    else:
      items = self.vocabulary.iteritems()
      
    for key, value in items:
      binary = np.zeros(self.size)
      binary[value] = 1
      self.binary_vocabulary[key] = binary
  
  def get_serialized_binary_representation(self):
    if len(self.binary_vocabulary) == 0:
      self.create_binary_representation()
      
    string = ""
    if sys.version_info >= (3,):
      items = self.binary_vocabulary.items()
    else:
      items = self.binary_vocabulary.iteritems()
    for key, value in items:
      array_as_string = np.array2string(value, separator = ',',\
        max_line_width = self.size * self.size)
      string += "{}{}{}\n".format(key, SEPARATOR, array_as_string[1:len(array_as_string)-1])
    return string
  
  def save(self, path, name = "words"):
    output_file_name = "{}/{}.vocab".format(path, name)
    with open(output_file_name, 'w') as f:
      f.write(self.get_serialized_binary_representation())
    
  def retrieve(self, path, name = "words"):
    with open("{}/{}.vocab".format(path, name), 'r') as dic_file:
      for line in dic_file:
        try:
          separator_position = line.index(SEPARATOR)
          key = line[:separator_position]
          value = line[separator_position + len(SEPARATOR):]
          value = np.fromstring(value, sep = ',')
        except:
          key = "\n"
          value = dic_file.readline()[len(SEPARATOR):]
          value = np.fromstring(value, sep = ',')
        self.binary_vocabulary[key] = value
        self.vocabulary[key] = np.where(value == 1)[0][0]
        self.token_lookup[np.where(value == 1)[0][0]] = key
  
    self.size = len(self.vocabulary)
    #print("dictionary retrieved from {}/{}.vocab".format(path, name))
    #print(self.vocabulary)