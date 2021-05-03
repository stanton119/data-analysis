import pandas as pd

def process(x
:
pd.DataFrame
) -> pd.DataFrame:
    y = pd.concat([x, x])
    return y


def process2(x  : pd.DataFrame, 
y : int
, z: bool 
= False) -> pd.DataFrame   :
    print('line_1')
    x_grouped = x.groupby(["col1"]).  loc[["col2", "col3"]].apply({'agg1':'mean','agg2':'median'})
    
    
    print(       'line_1')

    print('asd')
    return x_grouped, y
