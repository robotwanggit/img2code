from __future__ import print_function

import numpy as np
from Vocabulary import Vocabulary
import Utils
from Config import *
from Dataset import Dataset

class Generator(object):
  #Args:
  #  gui_paths and img_paths are two lists containing the token gui files and png/npz img files, respectively 
  @staticmethod
  def data_generator(voc, gui_paths, img_paths, batch_size, generate_binary_sequences = False, loop_only_one = False):
    assert len(gui_paths) == len(img_paths)
    voc.create_binary_representation()
    
    while 1:
      batch_input_images = []
      batch_token_segments = []
      batch_next_words = []
      sample_in_batch_counter = 0  #count how many samples have been added to a batch
      
      #process each gui-image files pair
      for i in range(0, len(gui_paths)):
        if img_paths[i].find(".npz") != -1:
          img = np.load(img_paths[i])["features"]
        else:
          img = Utils.get_preprocessed_img(img_paths[i], IMAGE_SIZE)
        gui = open(gui_paths[i], 'r')
        
        token_sequence = [START_TOKEN]
        for line in gui:
          line = line.replace(",", " , ").replace("\n", " \n ")
          tokens = line.split()
          for token in tokens:
            token_sequence.append(token)
        token_sequence.append(END_TOKEN)
        
        prefix = [PLACEHOLDER] * CONTEXT_LENGTH
        
        a = np.concatenate([prefix, token_sequence])
        
        #The number of training samples apppended for each gui-image file pair equals to N+2, in which N is the total
        #number of tokens in a gui file.
        for j in range(0, len(a) - CONTEXT_LENGTH):
          context = a[j: j+ CONTEXT_LENGTH]
          next_word = a[j+CONTEXT_LENGTH]
          
          batch_input_images.append(img)
          batch_token_segments.append(context)
          batch_next_words.append(next_word)
          sample_in_batch_counter += 1
          
          if sample_in_batch_counter == batch_size or (loop_only_one and i == len(gui_paths) -1):
            #print("Generating sparse vectors...")
            batch_next_words = Dataset.sparsify_labels(batch_next_words, voc)
            batch_token_segments = Dataset.token2index(batch_token_segments, voc, generate_binary_sequences)
            #print("convert arrays...")
            batch_input_images = np.array(batch_input_images)
            batch_token_segments = np.array(batch_token_segments)
            batch_next_words = np.array(batch_next_words)
            #print("Yield batch")
            yield ([batch_input_images, batch_token_segments], batch_next_words)
            
            batch_input_images = []
            batch_token_segments = []
            batch_next_words = []
            sample_in_batch_counter = 0
         