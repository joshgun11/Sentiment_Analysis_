from keras.layers import GlobalMaxPool1D,Embedding,Dense
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Bidirectional
from keras.layers import Dropout
from keras.models import Sequential
from keras import layers
import tensorflow as tf



class KModel():

    def __init__(self):

        return

    def baseline_model(self,embedding_dim ,embedding_matrix,num_classes,maxlen,vocab_size):
    
        
        model = Sequential()
        model.add(layers.Embedding(vocab_size,embedding_dim,weights = [embedding_matrix],
                           input_length=maxlen,trainable = True))
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.25))  
        model.add(Dense(16,activation = 'relu'))  
        model.add(Dense(num_classes,activation = "softmax")) 
        
    
        return model

    def model_cnn(self,embedding_dim,vocab_size,maxlen,num_classes,embedding_matrix):    
        model = Sequential()
        model.add(layers.Embedding(vocab_size,embedding_dim,weights = [embedding_matrix],
                           input_length=maxlen,trainable = True))
        model.add(layers.Conv1D(100, 5, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        

        return model


    def model_lstm(self,vocab_size,embedding_dim,maxlen,num_classes,embedding_matrix):

        model_lstm = Sequential()
        model_lstm.add(layers.Embedding(vocab_size,embedding_dim,weights = [embedding_matrix],
                           input_length=maxlen,trainable = True))
        model_lstm.add(SpatialDropout1D(0.5))
        model_lstm.add(Bidirectional(layers.LSTM(64))) 
        model_lstm.add(Dropout(0.5))
        model_lstm.add(Dense(16,activation = 'relu'))
        model_lstm.add(Dense(num_classes, activation='softmax'))
        

        return model_lstm

