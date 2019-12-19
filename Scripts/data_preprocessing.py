import pandas as pd

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
# Renaming of outcomes variables
outc_c_cols = ['Outcome_M01', 'Outcome_M02', 'Outcome_M03', 'Outcome_M04', 'Outcome_M05', 'Outcome_M06',
               'Outcome_M07', 'Outcome_M08', 'Outcome_M09', 'Outcome_M10', 'Outcome_M11', 'Outcome_M12'
               ]
#%%
# create intermediate DataFrame for time series
train_dates_outc_df = train_df[date_cols + outc_cols].rename(
    columns=dict(zip(outc_cols, outc_c_cols))
).reset_index().rename(         # Renaming variables for better visualization
    columns={
        'index': 'product',
        'Date_2': '0',
        'Date_1': 'D_OfRelease'}
)
#%%
# Creation of variable of difference between pre-release and release date
train_dates_outc_df['Delta_PreRel_Rel'] = train_dates_outc_df['D_OfRelease'] - train_dates_outc_df['0']

#%%
month = 30  # average number of days in a month
it = 1      # iterator for adding multiple months to starting date

# iteration to find dates for each outcome and renaming Outcomes variables to integers corresponding to months in year
for i in outc_c_cols:
    train_dates_outc_df[str(int(i[-2:]))] = train_dates_outc_df['0'] + it * month
    it += 1
#%%
# defining t0 value as 0, because sales started and only in the next 30 days the cumulative sales will be known
train_dates_outc_df['Outcome_M00'] = 0
#%%
# creation of time series DataFrame by using a tidy format (melt of pandas)
ts_train_df = pd.melt(
    train_dates_outc_df,
    ['product', 'D_OfRelease', 'Delta_PreRel_Rel'],
    [str(i) for i in range(0, 13)],
    'Month',
    'TimeValue'
)
#%%
# Cast Month variable as int
ts_train_df['Month'] = ts_train_df['Month'].astype('int8')
#%%
# adding the corresponding outcomes to its product and month
ts_train_df['Outcome'] = pd.melt(
    train_dates_outc_df,
    ['product', 'D_OfRelease'],
    ['Outcome_M00'] + outc_c_cols,
    'Month_Outcome',
    'Outcome'
)['Outcome']
#%%
# Sorting by Product and Month for debugging purposes
ts_train_df = ts_train_df.sort_values(by=['product', 'Month']).reset_index().drop('index', axis=1)
#%%
# Merge product characteristics
ts_train_df = pd.merge(
    ts_train_df,
    train_df[cat_cols + qnt_cols].reset_index().rename(
        columns={'index': 'product'}
    ),
    on='product'
)
#%%
# Sort by time value to have a running historical record
ts_train_df = ts_train_df.sort_values(by=['TimeValue'], ascending=True).reset_index().drop('index', axis=1)
#%%
# creating 2 distinct scenarios of NaN filling
# ts_train_df_zeros_version fills all NaN with 0
# ts_train_df_z_m_version fills NaN of Outcome variable with 0's and the rest with median
ts_train_df_zeros_version = ts_train_df.copy()
ts_train_df_zeros_version = ts_train_df_zeros_version.fillna(0)
ts_train_df_zeros_version = ts_train_df_zeros_version.astype('int32')

ts_train_df_z_m_version = ts_train_df.copy()
ts_train_df_z_m_version['Outcome'] = ts_train_df_z_m_version['Outcome'].fillna(0)
ts_train_df_z_m_version = ts_train_df_z_m_version.fillna(ts_train_df_z_m_version.median())
ts_train_df_z_m_version = ts_train_df_z_m_version.astype('int32')

