import pandas as pd


class DATA():
    def __init__(self):
        return

    def load_data(self,path):
        df = pd.read_csv(path)
    
        return df 


    def data_for_model(self,path):
        df = self.load_data(path)
        
        
        df.dropna(inplace = True)

        return df