#! user/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import sys
from os.path import basename
from img2code import img2code
import Utils
from Config import *
import numpy as np
from Sampler import Sampler

argv = sys.argv[1:]
if len(argv) < 4:
  print("Error: not enough argument supplied:")
  print("sample.py <trained weights dir> <trained model name> <input image> <output path> <search method (default: greedy)>")
  exit(1)
else:
  trained_weights_dir = argv[0]  #the directory contained the model structure and trained weight and the vocabulary, dataset meta info
  trained_model_name = "img2code" if argv[1]=="default_name" else argv[1]   #the name of the pretrained model
  input_img_path = argv[2]                                                 #path to the screenshot image as prediction input
  output_path = argv[3]                                                    #path to store the predicted token sequence gui file
  search_method = "greedy"

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_dir))
input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = img2code(input_shape, output_size, trained_weights_dir)
model.load(trained_model_name)
sampler = Sampler(trained_weights_dir, input_shape, output_size, CONTEXT_LENGTH)

file_name = basename(input_img_path)[:basename(input_img_path).find('.')]
predict_img = Utils.get_preprocessed_img(input_img_path, IMAGE_SIZE)
result, _ = sampler.predict_greedy(model, np.array([predict_img]))
print("Result greedy: {}".format(result))

with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
  out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))