#!usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import sys, os
from Config import *
import numpy as np
from Sampler import Sampler
from img2code import img2code
import Utils
from nltk.translate.bleu_score import corpus_bleu

argv = sys.argv[1:]
if len(argv) < 3:
  print("Error: not enough argument supplied:")
  print("evaluate.py <trained weights dir> <trained model name> <evaluation data dir>")
  exit(1)
else:
  trained_weights_dir = argv[0]  #the directory contained the model structure and trained weight and the vocabulary, dataset meta info
  trained_model_name = "img2code" if argv[1]=="default_name" else argv[1]   #the name of the pretrained model
  evaluation_data_dir = argv[2]  #directory containing the screenshot image and corresponding token gui files for evaluating
  
  meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_dir))
  input_shape = meta_dataset[0]
  output_size = meta_dataset[1]

  model = img2code(input_shape, output_size, trained_weights_dir)
  model.load(trained_model_name)
  sampler = Sampler(trained_weights_dir, input_shape, output_size, CONTEXT_LENGTH)
  
  reference = []
  predicted = []
  for f in os.listdir(evaluation_data_dir):
    if f.find(".gui") != -1:
      file_id = f[:-4]
      if os.path.isfile("{}/{}.npz".format(evaluation_data_dir, file_id)):
        img = np.load("{}/{}.npz".format(evaluation_data_dir, file_id))["features"]
      elif os.path.isfile("{}/{}.png".format(evaluation_data_dir, file_id)):
        img = Utils.get_preprocessed_img("{}/{}.png".format(evaluation_data_dir, file_id),IMAGE_SIZE)
      else:
        print("No image data found for gui file: {}".format(f))
        raise 
      result, _ = sampler.predict_greedy(model, np.array([img]))
      result = result.replace(START_TOKEN, "").replace(END_TOKEN, "")
      #print("P: "+result) #*******************************************************************
      predicted.append(result.split())
      file_name = f[:-4]
      with open("{}/{}.gui".format(evaluation_data_dir, file_name), 'r') as gui:
        context = gui.read()
        #print("A: "+context) #******************************
        reference.append([context.split()])
  bleu = corpus_bleu(reference, predicted)
  print(bleu)
