CONTEXT_LENGTH = 48     #time_steps in LSTM cell
IMAGE_SIZE = 256        #the shape of the screenshot img array will be [IMAGE_SIZE, IMAGE_SIZE, 3]
BATCH_SIZE = 64         #number of samples in a batch
EPOCHS = 50              #number of epochs to train on the whole sample data

START_TOKEN = "<start>" #a special token to indicate the start of token sequences in a new gui file
END_TOKEN = "<end>"     #a special token to indicate the end of token sequences in a gui file
PLACEHOLDER = " "       #a special token to represent the null context
