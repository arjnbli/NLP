# Text Generation Using LSTMs

LSTM networks can be trained on a corpus of text to build a language model.This allows us to predict the next
word in a sequence, given the words that precede it. This is a Keras implementation of a character level LSTM 
language model. 

The corpus of text on which the LSTM is trained is a string. A vocabulary of the characters occuring in the string
is constructed and each character is assigned an index. Using this scheme, a string of characters is transformed 
into a sequence of integers. The training sequences are constructed by sliding a window of size 100 across the corpus of text.
Starting from the beginning of the corpus of text, the first 100 characters are used as training input and the 101st character 
is used as the training output. In this manner, the sliding window is moved across the corpus of text such that each training 
example consists of a training input of length 100 and, the training output consists of a single integer corresponding to the 
index of the next character in the sequence. It must be noted that the last 100 characters in the text corpus are not used for 
training as there is no character in the sequence after that. The output labels are encoded as one hot vectors.

A many-to-one LSTM network with 128 hidden units is used. The LSTM output is passed to a fully connected layer and
the categorical cross entropy function is used as the optimization objective. The model is trained for desired number
of epochs and the weights are saved after each epoch.

Once the network has been trained, the weights corresponding to the minimum loss are loaded. The text is generated
by passing a random sequence from the text corpus as input. The generated output is appended to the input sequence and
the first term is removed. This new sequence is then used as input to generate the next character and this process
continues for desired number of iterations.

There are two files - lstm_language_model.py and lstm_text_gen.py. 

lstm_language_model.py trains the lstm on a corpus of text and saves weights after each epoch. The path to the corpus of 
text needs to provided.

lstm_text_gen.py loads weights and generates the sequence. The path to the location where the weights are stored needs to
be provided.


