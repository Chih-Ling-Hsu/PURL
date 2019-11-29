import pandas as pd

class Discretizer():
    def __init__(self, attributeList, bin_count=8, style='equal-sized'):
        self.attributes = attributeList
        self.bin_count = bin_count
        self.style = style

    def fit_transform(self, df):
        for i in range(len(self.attributes['name'])):            
            att = self.attributes['name'][i]
            _type = self.attributes['type'][i]
            if _type == 'continuous':
                if self.style == 'equal-depth':
                    df[att], self.bins = pd.qcut(df[att], q=self.bin_count, duplicates='drop', retbins=True)
                elif self.style == 'equal-sized':
                    df[att], self.bins = pd.cut(df[att], bins=self.bin_count, duplicates='drop', retbins=True)
                df[att] = df[att].astype(str)
        return df
    
    def transform(self, df):
        for i in range(len(self.attributes['name'])):            
            att = self.attributes['name'][i]
            _type = self.attributes['type'][i]
            if _type == 'continuous':
                df[att] = pd.cut(df[att], bins = self.bins).astype(str)                        
        return df