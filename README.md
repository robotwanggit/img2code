# img2code
A neural network trained to generate html codes from webpage screenshots.

This repository is an implementation of Tony Beltramelli's paper, [pix2code: Generating Code from a Graphical User Interface Screenshot](https://arxiv.org/abs/1705.07962).   
Some codes here are borrowed from Tony's opensourced codes in [this](https://github.com/tonybeltramelli/pix2code.git) repository.  
Compared to Tony's version, evaluation with BLEU score is added to evaluate the accuracy of training result. Besides of one-hot encoding of tokens in the original implementation, word embeddings are tried in the model to better represent the relation between tokens.

## Folder structures
``` bash
|-one-hot version           #use one-hot encoding of tokens in the model
  |-Compiler                #Folder containing the compiler which compiles token sequence into html codes (from pix2code)
  |-datasets                #Folder storing the training dataset. A light-weight training dataset containing 100 image-token file pairs are provided. 
  |-images                  #Folder containing the screenshot images for test  
  |-evaluation              #Folder containing evaluation dataset.  
  |-trained model           #Folder to store the trained model.  
  |  |-img2code.h5          #model weights saved as {model_name}.h5 after training  
  |  |-img2code.json        #model structure saved as {model_name}.json after training    
  |  |-words.vocab          #vocabulary built from token files  
  |  |-meta_datasets.npy    #model metadata including input_shape, output_shape, etc  
  |-...                     #source codes implementing img2code model  
|-embedding version         #use word-embedding of tokens in the model
    |-...                   #folder structure same as one-hot version
```
## Model Structure
The img2code model is composed of an encoder and decoder. The encoder contains CNN and LSTM networks in parrallel, which encodes into features the webpage screenshot image and tokens sequences. The features of image/token inputs are then concatenated together and provide into the decoder.  
The decoder is an LSTM network, which infers tokens one by one.   
In the training stage, the screenshot image and the corresponding token sequences are both provided to train the model. In the prediction stage, only webpage image is needed. The generated token sequences can be compiled into html codes by the Compiler.
The structure of img2code model using word embedding is as below
<p align="center"><img src="/structure.jpg?raw=true" width="400px"></p>

## Train model
```
cd img2code/embedding version

#provide input dir containing training data and output dir to save trained model and metadata
#provide non-zero number for use_generatator if training data are too large to be loaded into memory all at once
#provide the saved weight file in previous training if you don't want to train from scratch
train.py <input_dir> <output_dir> <use_generator:optional> <pretrained_model:optional>
```
A light-weight training dataset containing 100 image-token file pairs are provided. The full datasets containing 1500 training pairs can be downloaded [here](https://github.com/tonybeltramelli/pix2code.git)  
Using Tesla K80 GPU, it takes about 210 s to train one epoch on the light-weight dataset with batch size 64 and context size 48.
## Evaluate training result
```
#provide the <trained weight dir> which contains the saved model. Usually same as <output_dir> in the training stage
#provide the model_name, currently only "default_name" is supported
#provide <evaluation data dir> with the folder that stores the evaluation image-token file pairs
evaluate.py <trained weights dir> default_name <evaluation data dir>
```
After trained for 120 epochs, both one-hot version and embedding version shows loss around 0.01. The bleu score for one-hot version and embedding version are 0.31 and 0.32, respectively.  
Trained with a small amount of data, the model must suffer from severe over-fitting problem. The performance is expected be further improved if trained with large dataset.

## Convert webpage image to html codes
```
#first step: generate token sequence file based on the webpage screenshot imge
#provide the <trained weight dir> which contains the saved model. Usually same as <output_dir> in the training stage
#provide the model_name, currently only "default_name" is supported
#provide the file path to the website screenshot image file
#provide the directory to store the generated token file
sample.py <trained weights dir> default_name <input image> <output path>

#second step: compile the token sequence file to html file
#provide the directory containing the token sequence files, usually same as <output path> in the first step
./compiler/web-compiler-drive.py <dir containing token files>
```
