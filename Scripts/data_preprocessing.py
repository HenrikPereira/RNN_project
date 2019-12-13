import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

test_df = pd.read_csv(r'./Data/TestDataset.csv')
train_df = pd.read_csv(r'./Data/TrainingDataset.csv')

cat_cols = []
qnt_cols = []
date_cols = []
outc_cols = []
for col in train_df.columns:
    if col.startswith('Cat'):
        cat_cols.append(col)
    elif col.startswith('Quan'):
        qnt_cols.append(col)
    elif col.startswith('Date'):
        date_cols.append(col)
    elif col.startswith('Outcome'):
        outc_cols.append(col)


def drop_empty_cols(dataframe):
    """

    :param dataframe:
    :return:
    """
    cols = dataframe.columns.tolist()

    noneed = 0
    list_noneed = []

    for c in cols:
        if len(dataframe[c].unique()) == 1:
            dataframe.drop(columns=[c], inplace=True)
            objects.remove(c)
            noneed += 1
            list_noneed.append(c)

    if noneed == 1:
        print('\tThe {0} column was droped.'.format(list_noneed))
    elif noneed > 1:
        print('\t{0} columns, {1} were droped.'.format(noneed, list_noneed))
    else:
        print('\tNo columns were removed.')
