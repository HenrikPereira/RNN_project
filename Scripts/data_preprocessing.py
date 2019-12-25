import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

#%%
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
#%%
# Fill of nan values of Pre-release with the release date
train_df['Date_2'].fillna(train_df['Date_1'], inplace=True)

#%%
# List to rename outcomes variables
outc_c_cols = ['Outcome_M01', 'Outcome_M02', 'Outcome_M03', 'Outcome_M04', 'Outcome_M05', 'Outcome_M06',
               'Outcome_M07', 'Outcome_M08', 'Outcome_M09', 'Outcome_M10', 'Outcome_M11', 'Outcome_M12']

#%%
# create intermediate DataFrame for time series
intermediate_df = train_df[date_cols + outc_cols].rename(
    columns=dict(zip(outc_cols, outc_c_cols))
).reset_index().rename(         # Renaming variables for better visualization
    columns={
        'index': 'product',
        'Date_2': 'D_PreRelease',
        'Date_1': 'D_Release'}
)
#%%
# Reducing data values to months and Creation of variable of difference between pre-release and release date
month = 30  # average number of days in a month

intermediate_df['D_Release'] = intermediate_df['D_Release'] // month
intermediate_df['D_PreRelease'] = intermediate_df['D_PreRelease'] // month
intermediate_df['Delta_PreRel_Rel'] = intermediate_df['D_Release'] - intermediate_df['D_PreRelease']


#%%
it = 1      # iterator for adding multiple months to starting date

# iteration to find dates for each outcome and renaming Outcomes variables to integers corresponding to months in year
for i in outc_c_cols:
    intermediate_df[str(int(i[-2:]))] = intermediate_df['D_Release'] + it
    it += 1

# the following value will be use to cut off the extra dates not currently in the original dataset
max_release_date = intermediate_df['D_Release'].max()

#%%
# Processing of variables to match further on
# The prerelease data as an time0 marker
intermediate_df.rename(columns={'D_PreRelease': '0'}, inplace=True)

# Drop of the release data as it has already been used to generate the time series
intermediate_df.drop('D_Release', axis=1, inplace=True)

# Creation of a marker (negative) of the average mean of outcome for all products to signalize the NN that something
# happened at this time, and not related to the future outcome itself but something that can influence the contemporary
# products at that time
intermediate_df['Marker_PreRelease_M00'] = -2000

# The products that had no prerelease date would have a null marker
for i in intermediate_df[intermediate_df['Delta_PreRel_Rel'] == 0].index:
    intermediate_df.loc[i, 'Marker_PreRelease_M00'] = np.nan

#%%
# creation of time series DataFrame by using a tidy format (melt of pandas)
ts_train_df = pd.melt(
    intermediate_df,
    ['product'],
    [str(i) for i in range(0, 13)],
    'Product_Month',
    'TimeValue'
)
#%%
# Cast Month variable as int
ts_train_df['Product_Month'] = ts_train_df['Product_Month'].astype('int8')

# Cast TimeValue as int
ts_train_df['TimeValue'] = ts_train_df['TimeValue'].astype('int16')

#%%
# adding the corresponding outcomes to its product and month
ts_train_df['Outcome'] = pd.melt(
    intermediate_df,
    ['product'],
    ['Marker_PreRelease_M00'] + outc_c_cols,
    'Month_Outcome',
    'Outcome'
)['Outcome']

#%%
# Creating a time series dataframe with pivoted outcomes (one column for each product)
pts_train_df = ts_train_df.pivot('TimeValue', 'product', 'Outcome')

# Removal of the extra months created for the time series which had no data (last ones)
pts_train_df = pts_train_df[pts_train_df.sum(axis=1) != 0]

#%%
# Create a product dataframe
product_train_df = train_df[cat_cols + qnt_cols]

# Same treatment expected for the test DataFrame...

#%%
# creating 2 distinct scenarios of NaN filling

# product_train_df_mVersion fills all NaN of the products with the variables median
# product_train_df_zVersion fills all NaN of the products with zeroes
product_train_df_mVersion = product_train_df.copy().fillna(product_train_df.median())
product_train_df_zVersion = product_train_df.copy().fillna(0)

# pts_train_df_zVersion fills all NaN of the time series with zeros
pts_train_df_zVersion = pts_train_df.copy().fillna(0)

#%%
# Deletion of non essential variables
del cat_cols, col, date_cols, i, intermediate_df, max_release_date, month, outc_c_cols, outc_cols, product_train_df, \
    train_df, qnt_cols, it

# #%%  -------------> STAND BY CODE <---------------
# # Merge product characteristics
# ts_train_df_final = pd.merge(
#     ts_train_df,
#     product_train_df.reset_index().rename(
#         columns={'index': 'product'}
#     ),
#     on='product'
# )
#
# # Sort by time value to have a running historical record
# ts_train_df_final = \
#   ts_train_df_final.sort_values(by=['TimeValue'], ascending=True).reset_index().drop('index', axis=1)
#
# #%%
# # creating 2 distinct scenarios of NaN filling
# # ts_train_df_zeros_version fills all NaN with 0
# # ts_train_df_z_m_version fills NaN of Outcome variable with 0's and the rest with median
# ts_train_df_zeros_version = ts_train_df_final.copy()
# ts_train_df_zeros_version = ts_train_df_zeros_version.fillna(0)
# ts_train_df_zeros_version = ts_train_df_zeros_version.astype('int32')
#
# ts_train_df_z_m_version = ts_train_df_final.copy()
# ts_train_df_z_m_version['Outcome'] = ts_train_df_z_m_version['Outcome'].fillna(0)
# ts_train_df_z_m_version = ts_train_df_z_m_version.fillna(ts_train_df_z_m_version.median())
# ts_train_df_z_m_version = ts_train_df_z_m_version.astype('int32')
