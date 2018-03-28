  #preprocess data in input_path
  #the data in input_path is image file storing the screenshot of webpages
  #and gui files storing the DSL(domain specific language) of tokens which
  #can be compiled into html language
  
  #The webpage screenshots are resized and normalized into numpy array with
  #shape [256, 256, 3]
  
  #The vocabulary is built based on tokens contained in all the gui files
  #plus the <start>, <end> and <placeholder> tokens.
  #Then the tokens in each gui file are converted to one-hot coding and the
  #embedding vectors are learned.
  
  #------------------------------------------------------------------------#
  #The model structure is composed of an encoder and decoder. The encoder
  #contains CNN and LSTM network in parrallel, which encodes into features
  #the screenshot and the DSL tokens, respectively. The featurs of screenshot
  #and DSL tokens are concatenated together.
  #The decoder is composed of an LSTM network, which inference the token
  #based on the input features.
  
  #In one step of training, the numpy array corresponding to the screenshort
  #and the embedding vectors corresponding to a segment of tokens are provided
  #as inputs. The input is encoded into features by the encoder and then
  #decoded into a token representing the most probable next token following
  #the input segment of tokens. The infered next token is then concatenated
  #into the input tokens and the updated input token segment is provided
  #into the model together with the screenshot array. A new step of training
  #start and the training continued until the prescribe number of epoches
  #finishes.
  
Update: 
  support generator mode for large training datasets
  add evaluation module to measure the accuracy