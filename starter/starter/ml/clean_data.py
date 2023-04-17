import pandas as pd
import numpy as np


def basic_cleaning(df, output_path, target, test=False):
    '''
    Basic cleaning of data
    '''

    #remove spaces from column names
    df.columns = df.columns.str.replace(" ", "")

    #filter categorical columns and numerical columns:
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    
     #replacing spaces & - with underscore in categorical columns:
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

    #replacing ? with nan:
    df = df.replace('?', np.nan)

    #fill nan with mode for categorical columns:
    for col in cat_cols:
        if col != target:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            cat_cols.remove(col)

    #fill nan with mean for numerical columns:
    for col in num_cols:
        if col != target:
            df[col] = df[col].fillna(df[col].mean())
        else:
            num_cols.remove(col)

    #save cleaned data:
    if test==False:
        try:
            df.to_csv(output_path, index=False)
            return df, cat_cols, num_cols
        except:
            pass
    else:
        return df, cat_cols, num_cols