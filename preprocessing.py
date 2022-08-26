import re
from sklearn import preprocessing
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import os 
from parser import KParseArgs
import pickle


class PreProcess():
    def __init__(self):
        return

    def clean_text(self,x):
        pattern = r'[^a-zA-z0-9\s]'
    
        text = re.sub(pattern, '', x)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def clean_numbers(self,x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]**{4}**', '####', x)
            x = re.sub('[0-9]**{3}**', '###', x)
            x = re.sub('[0-9]**{2}**', '##', x)
        return x

    def prepare_text(self,corpus):
        cleaned_text = []
        for text in corpus:
            text = self.clean_text(text)
            cleaned_text.append(text)
        return cleaned_text


    def prepare_targets(self,y_train, y_test):
        parser = KParseArgs()
        args = parser.parse_args()
        le = preprocessing.LabelEncoder()

        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        y_train_enc = tf.one_hot(y_train_enc, args.num_classes) 
        y_test_enc = tf.one_hot(y_test_enc, args.num_classes) 
        return y_train_enc, y_test_enc

    def make_sequences(self,X_train,X_test,maxlen):

        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_train_padded = pad_sequences(X_train_seq, padding='post', maxlen=maxlen,truncating="post")
        X_test_padded = pad_sequences(X_test_seq, padding='post', maxlen=maxlen,truncating="post") 

        return X_train_padded,X_test_padded,tokenizer


    def create_embedding_matrix(self,filepath, word_index, embedding_dim,vocab_size):
    
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        file = open(filepath, errors = 'ignore', encoding='utf8')
        for line in file:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.asarray(vector, dtype=np.float32)[:embedding_dim]

        return embedding_matrix







