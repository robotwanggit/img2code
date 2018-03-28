from __future__ import absolute_import

from keras.layers import Input, Dense, Dropout, RepeatVector, LSTM, \
                         Conv2D, MaxPooling2D, Flatten, concatenate, Embedding
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import RMSprop
from Config import *

class img2code(object):
  #input_shape, the shape of the tensor representing the webpage screenshot
  #output_size, dimension of  vocabulary size. 
  #output_path, oath to store the built Vocabulary, to save model structure and weights
  #name, the name of the model, "img2code" in default
  def __init__(self, input_shape, output_size, output_path, name = "img2code"):
    self.name = name
    self.image_shape = input_shape
    self.output_path = output_path
    
    image_encoder = Sequential()
    image_encoder.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'valid', 
        data_format = 'channels_last',activation = 'relu', input_shape = input_shape))
    image_encoder.add(Conv2D(32, (3, 3), padding = 'valid', activation = 'relu'))
    image_encoder.add(MaxPooling2D(pool_size = (2, 2)))
    image_encoder.add(Dropout(0.25))
    
    image_encoder.add(Conv2D(64, (3, 3), padding = 'valid', activation = 'relu'))
    image_encoder.add(Conv2D(64, (3, 3), padding = 'valid', activation = 'relu'))
    image_encoder.add(MaxPooling2D(pool_size = (2, 2)))
    image_encoder.add(Dropout(0.25))
    
    image_encoder.add(Flatten())
    image_encoder.add(Dense(1024, activation = 'relu'))
    image_encoder.add(Dropout(0.3))
    image_encoder.add(Dense(1024, activation = 'relu'))
    image_encoder.add(Dropout(0.3))
    image_encoder.add(RepeatVector(CONTEXT_LENGTH))
    
    token_encoder = Sequential()
    token_encoder.add(Embedding(output_size, EMBEDDING_SIZE, input_length = CONTEXT_LENGTH))
    token_encoder.add(
      LSTM(units = 128, activation = 'tanh', recurrent_activation = 'hard_sigmoid',
        dropout = 0.0, recurrent_dropout = 0.0, implementation = 2, return_sequences = True,
        stateful = False))

    token_encoder.add(
      LSTM(units = 128, activation = 'tanh', recurrent_activation = 'hard_sigmoid',
        dropout = 0.0, recurrent_dropout = 0.0, implementation = 2, return_sequences = True,
        stateful = False)
    )
    
    image_input = Input(shape = input_shape)
    encoded_image = image_encoder(image_input)
    token_input = Input(shape = (CONTEXT_LENGTH, ))
    encoded_token = token_encoder(token_input)
    
    decoder = concatenate([encoded_image, encoded_token])
    decoder = LSTM(512, return_sequences = True)(decoder)
    decoder = LSTM(512, return_sequences = False)(decoder)
    decoder = Dense(output_size, use_bias = True, activation = 'softmax')(decoder)
    
    self.model = Model(inputs = [image_input, token_input], outputs = decoder)
    optimizer = RMSprop(lr = 0.0001, clipvalue = 1.0)
    self.model.compile(loss = 'categorical_crossentropy', optimizer =  optimizer)
  
  def fit(self, images, token_segments, next_words):
    self.model.fit([images, token_segments], next_words, shuffle = False, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1)
    self.save()
   
  def fit_generator(self, generator, steps_per_epoch):
    self.model.fit_generator(generator, steps_per_epoch = steps_per_epoch, epochs = EPOCHS, \
      use_multiprocessing = False, verbose = 1)
    self.save()
    
  def predict(self, image, token_segment):
    return self.model.predict([image, token_segment], verbose = 0)[0]
    
  def predict_batch(self, images, token_segments):
    return self.model.predict([images, token_segments], verbose = 1)
   
  def load(self, name = ""):
    output_name = self.name if name == "" else name
    with open("{}/{}.json".format(self.output_path, output_name)) as json_file:
      loaded_model_json = json_file.read()
    self.model = model_from_json(loaded_model_json)
    self.model.load_weights("{}/{}.h5".format(self.output_path, self.name))
    
  def save(self):
    model_json = self.model.to_json()
    with open("{}/{}.json".format(self.output_path, self.name), 'w') as json_file:
      json_file.write(model_json)
    self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))
    