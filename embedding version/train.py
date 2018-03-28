#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import sys
from img2code import img2code
from Dataset import Dataset
from Vocabulary import Vocabulary
from Generator import Generator
import numpy as np
from Config import *

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#parameters:
#input_path: path containing the training data, including screenshot image/npz file
#            and token gui file
#output_path: path to save the composed dictionary, the model
#            structure and weights
#pretrained_model: file of a pretrained model weight data
def run(input_path, output_path, use_generator = False, pretrained_model = None):
  """Train the modal with training data in input_path and store the trained model in output_path"""
  np.random.seed(1234)
  dataset = Dataset()
  dataset.load(input_path, generate_binary_sequences = False)
  dataset.save_metadata(output_path)
  dataset.voc.save(output_path)
  assert len(dataset.input_images) == len(dataset.token_segments) == len(dataset.next_words)
  
  input_shape = dataset.input_shape
  output_size = dataset.output_size
  model = img2code(input_shape, output_size, output_path)
  
  if pretrained_model is not None:
    model.model.load_weights(pretrained_model)
  
  if not use_generator:
    dataset.convert_arrays()
    print("total training samples: {}".format(len(dataset.input_images)))
    print("imgage data shape: {}, tokens data shape: {} and next token shape: {}"\
      .format(dataset.input_images.shape, dataset.token_segments.shape, dataset.next_words.shape))  
    model.fit(dataset.input_images, dataset.token_segments, dataset.next_words)
  else:
    gui_paths, img_paths = Dataset.load_paths_only(input_path)
    steps_per_epoch = dataset.size / BATCH_SIZE
    voc = Vocabulary()
    voc.retrieve(output_path)
    generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size = BATCH_SIZE, generate_binary_sequences=False)
    model.fit_generator(generator, steps_per_epoch = steps_per_epoch)
if __name__ == "__main__":
  argv = sys.argv[1:]
  
  if len(argv) < 3:
    print("Error: not enough argument supplied:")
    print("train.py <input_path> <output_path> <use_generator:optional> <pretrained_path:optional>")
    exit(1)
  else:
    input_path = argv[0]
    output_path = argv[1]
    use_generator = False if argv[2]=="0" else True
    pretrained_weights = None if len(argv)<4 else argv[3]
   
    
  run(input_path, output_path, use_generator = use_generator, pretrained_model = pretrained_weights)
  