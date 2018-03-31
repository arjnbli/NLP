import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def load_data(path):
    """
    Loads corpus of text
    
    Argument:
    path -- string,path to the corpus of text
 
    Returns:
    raw_data -- string containing entire corpus of text
   
    chars -- list containing all the distinct characters
             in the given corpus of text
    
    char_dict -- dictionary with all the distinct characters
                 in given corpus of text mapped to integers
    
    idx_dict -- dictionary with integers mapped back to the
                characters they represent
    
    total_characters -- total characters in given corpus of text
    
    vocabulary_size -- number of unique characters in given
                       corpus of text
    
    """
    raw_data=open(path,'r',encoding='utf8').read()
    raw_data=raw_data.lower()
    
    chars=sorted(list(set(raw_data)))
    char_dict={char:idx for (idx,char) in enumerate(chars)}
    idx_dict={idx:char for (idx,char) in enumerate(chars)}
    
    
    total_characters=len(raw_data)
    vocabulary_size=len(chars)
    
    return raw_data,chars,char_dict,idx_dict,total_characters,vocabulary_size


def training_data(raw_data,char_dict,seq_length,total_characters,vocabulary_size):
    """
    Converts raw training data into numpy arrays which are used as inputs
    to an LSTM network

    Arguments:
    raw_data -- string containing entire corpus of text

    char_dict -- dictionary with all the distinct characters
                 in given corpus of text mapped to integers

    seq_length -- int, length of training sequences

    total_characters -- int, total characters in given
                        corpus of text

    vocabulary_size -- number of unique characters in given
                       corpus of text

 
    Returns:
    x -- list of input training sequences

    y -- list of outputs corresponding to input
         training sequences

    X -- numpy array with normalized input training
         sequences

    y -- numpy array containing one-hot encoded
         output labels

    training_patterns -- total characters in given corpus of text

    vocabulary_size -- number of unique characters in given
                       corpus of text
    """
    x=[]
    y=[]
    for i in range(0,total_characters-seq_length,1):
        input_seq=raw_data[i:i+seq_length]
        out_seq=raw_data[i+seq_length]
        x.append([char_dict[char] for char in input_seq])
        y.append(char_dict[out_seq])
    training_patterns=len(x)
    X = np.reshape(x, (training_patterns, seq_length, 1))
    X = X / float(vocabulary_size)
    Y = np_utils.to_categorical(y)
    return X,Y,x,y,training_patterns

def text_model(x,y):
    """
	 Creates LSTM model

	 Arguments:
	 x -- numpy array containing input sequences
	 y -- numpy array containing output labels

	 Returns:
	 model -- keras model
    """
    text_input = Input(shape=(x.shape[1],x.shape[2]))
    lstm = LSTM(128)(text_input)
    dense=Dense(y.shape[1],activation='softmax')(lstm)
    model = Model(inputs=text_input, outputs=dense)
    return model


#Training corpus is loaded from path
text,chars,char_to_idx,idx_to_dict,n_characters,n_vocabulary=load_data('path')
print('Total number of characters are ',n_characters)
print('Vocabulary size is ',n_vocabulary)

#Training data is created
X,Y,_,_,n_patterns=training_data(text,char_to_idx,100,n_characters,n_vocabulary)
print('Total number of training patterns are ',n_patterns)

#Instance of LSTM model defined
model=text_model(X,Y)
#Model is set up with crossentropy as loss function and adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')
#Checkpoint is defined
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#Model is trained using the fit method
model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)