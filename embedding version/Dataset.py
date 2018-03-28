from __future__ import print_function
import os

from Vocabulary import *
from Config import *
import Utils

class Dataset(object):

  #Intialize members in class Dataset
  #input_shape is the np.array shape of the precessed screenshot img, 
  #which should equal to [IMAGE_SIZE, IMAGE_SIZE, 3]
  #output_size is the dimension of vocabulary size
  #ids stores the file id corresponding to each training sample. len(ids) equals to number of training samples
  #input_images stores resized and normalized image array of input_shape. len(input_images) equals to number of traning samples
  #token_segments stores segments of token index or segments of token in one-hot format. A segment has length of CONTEXT_LENGTH, 
  #len(token_segments) equals to number of training samples
  #next_words stores the next_words in one-hot format, each corresponds to the next word following the token segment in token_segments
  def __init__(self):
    self.input_shape = None
    self.output_size = None
    self.ids = []
    self.input_images = []
    self.token_segments = []
    self.next_words = []
    
    self.voc = Vocabulary()
    self.size = 0
    
  @staticmethod
  def load_paths_only(path):
    print("Parsing data...")
    token_paths = []
    img_paths = []
    for f in os.listdir(path):
      ext_pos = f.find(".gui")
      if ext_pos != -1:
        token_paths.append(path + "/" + f)
        file_id = f[:ext_pos]
        
        if os.path.isfile("{}/{}.png".format(path, file_id)):
          img_paths.append("{}/{}.png".format(path, file_id))
        elif os.path.isfile("{}/{}.npz".format(path, file_id)):
          img_paths.append("{}/{}.npz".format(path, file_id))
        else:
          print("No image file found for {}".format(f))
          raise
    assert len(token_paths) == len(img_paths)
    return token_paths, img_paths
  
  #Load data from the provided path, the img data are retrived from the png/npz file and token data from the gui file.
  def load(self, path, generate_binary_sequences = False):
    print("Loading data...")
    for f in os.listdir(path):
      if f.find(".gui") != -1:
        with open("{}/{}".format(path, f), 'r') as gui:
          file_id = f[:f.find(".gui")]
          if os.path.isfile("{}/{}.png".format(path, file_id)):
            img = Utils.get_preprocessed_img("{}/{}.png".format(path, file_id),IMAGE_SIZE)
          elif os.path.isfile("{}/{}.npz".format(path, file_id)):
            img = np.load("{}/{}.npz".format(path, file_id))["features"]
          #append the sample_id, img and token_segment and next_word to class Dataset members
          self.append(file_id, gui, img)
    
    #truncate ids, input_images, token_segments and next_words to have multiple length of BATCH_SIZE
    multiple_num = int(len(self.ids)/BATCH_SIZE) * BATCH_SIZE
    self.ids = self.ids[:multiple_num]
    self.input_images = self.input_images[:multiple_num]
    self.token_segments = self.token_segments[:multiple_num]
    self.next_words = self.next_words[:multiple_num]
    
    print("Generating sparse vectors...")
    self.voc.create_binary_representation()
    #convert self.next_words from format string to format one-hot encoding index
    self.next_words = self.sparsify_labels(self.next_words, self.voc)
    #convert self.token_segments from format string to format index
    self.token_segments = self.token2index(self.token_segments, self.voc, generate_binary_sequences)
    
    self.size = len(self.ids)
    assert self.size == len(self.input_images) == len(self.token_segments) == len(self.next_words)
    assert self.voc.size == len(self.voc.vocabulary)
    
    print("Dataset size: {}".format(self.size))
    print("Vocabulary size: {}".format(self.voc.size))

    self.input_shape = self.input_images[0].shape
    self.output_size = self.voc.size

    print("Input shape: {}".format(self.input_shape))
    print("Output size: {}".format(self.output_size))
  

  #append the sample_id, img and token_segment and next_word to class Dataset members
  #The number of training samples apppended when calling append equals to N+2, in which N is the total
  #number of tokens of the token sequence in a gui file.
  #parameters:
  #gui: string containing the token sequence of a gui file
  #img: resized and normlized numpy array represent the screenshot img
  #sample_id: hash id representing a specific screenshot
  def append(self, sample_id, gui, img, to_show = False):
    if to_show:
      pic = img*255
      pic = np.array(pic, dtype = np.uint8)
      Utils.show(pic)
      
    token_sequence = [START_TOKEN]
    for line in gui:
      line = line.replace(",", " ,").replace("\n", " \n")
      tokens = line.split(" ")
      for token in tokens:
        self.voc.append(token)
        token_sequence.append(token)
    token_sequence.append(END_TOKEN)
    
    prefix = [PLACEHOLDER] * CONTEXT_LENGTH
    
    a = np.concatenate([prefix, token_sequence])
    for j in range(0, len(a) - CONTEXT_LENGTH):
      context = a[j: j+ CONTEXT_LENGTH]
      next_word = a[j+CONTEXT_LENGTH]
      self.ids.append(sample_id)
      self.token_segments.append(context)
      self.input_images.append(img)
      self.next_words.append(next_word)
      
  def convert_arrays(self):
    self.input_images = np.array(self.input_images)
    self.token_segments = np.array(self.token_segments)
    self.next_words = np.array(self.next_words)
  
  #convert the token_segments from list of sequences of token string to
  #list of sequences of numbers, the number may be index in the vocabulary
  #or one-hot encoding format of the token in the vocabulary
  @staticmethod
  def token2index(token_segments, voc, binarize = False):
    temp = []
    for token_segment in token_segments:
      sparse_vectors_sequence = []
      for token in token_segment:
        if binarize:
          sparse_vectors_sequence.append(voc.binary_vocabulary[token])
        else:
          sparse_vectors_sequence.append(voc.vocabulary[token])
      temp.append(sparse_vectors_sequence)
    return temp
  
  @staticmethod
  def sparsify_labels(next_words, voc):
    temp = []
    for label in next_words:
      temp.append(voc.binary_vocabulary[label])
    return temp
    
  def save_metadata(self, path):
    np.save("{}/meta_dataset".format(path), np.array([self.input_shape, self.output_size, self.size]))