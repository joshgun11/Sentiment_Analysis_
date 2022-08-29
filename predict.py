from tensorflow import keras
import pickle 
import pandas as pd 
from preprocessing import PreProcess
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from parser import KParseArgs
import sys


class Predict():
    def __init__(self) -> None:
        pass

    def load_model(self,model_path,tokenizer_path):
        
        model = keras.models.load_model(model_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return model,tokenizer
    
    def predict(self,args):
        process = PreProcess()
        
        df = pd.read_csv('train_data.csv')
        target =  df['label']
        model,tokenizer = self.load_model(args.model_path,args.tokenizer_path)
        text = args.text
        clean_text = process.clean_text(text)
        clean_text = process.clean_numbers(clean_text)


        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=50)
        pred = model.predict(padded)
        predicted_class = pred.argmax(axis=-1)
   
        classes  = {'0':'Irrelevant', '1':'Negative', '2':'Neutral', '3':'Positive'}
        actual_class = classes[str(predicted_class[0])]
        return actual_class


if __name__ == '__main__':

    parser = KParseArgs()
    args = parser.parse_args()
    flag = len(sys.argv) == 1
    result = Predict().predict(args)
    print(result)

