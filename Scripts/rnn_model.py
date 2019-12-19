import numpy as np
import data_preprocessing
import tensorflow.keras as k
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import MinMaxScaler

# %%
# Loading Data as Pandas DataFrame
train_df = data_preprocessing.ts_train_df_z_m_version

# MinMax Scaling if necessary
# this transformation can contaminate data along the whole dataset... use with caution...
minmax = MinMaxScaler()
# %%
len(np.array(train_df))


# %%
# Adapted from prof Mafalda Challenge Notebook


def data_generator(
        data,  # input data
        lookback,  # how many timesteps back the input should go
        delay,  # how many timesteps in the future the target should be
        min_index,  # index in the data array that delimits which timesteps to draw from
        max_index,  # index in the data array that delimits which timesteps to draw from
        batch_size,  # the number of samples per batch
        shuffle=False,  # whether to shuffle the samples or draw them in chronological order
        step=1  # the period, in timesteps, at which data is sampled
):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))

        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


def make_mode(
        lr=0.001
):
    model = Sequential()
    model.add([
        Embedding(
            input_dim=10,
            output_dim=10,
        ),
        SimpleRNN(
            10,
            activation='relu',
        ),
        Dropout(0.2),
        k.layers.BatchNormalization(),
        Dense(1)
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    return model
