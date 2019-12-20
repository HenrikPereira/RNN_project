import numpy as np
import data_preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, save_model, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Embedding, Input, BatchNormalization, Concatenate, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import he_uniform, he_normal
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %%
# Loading Data as Pandas DataFrame
train_df = data_preprocessing.ts_train_df_z_m_version

# %%
# MinMax Scaling if necessary
# this transformation can contaminate data along the whole dataset... use with caution...
minmax = MinMaxScaler()

# %%
train_feat = np.array(train_df.iloc[:, 6:].values).astype('int32')
train_feat_labels = np.array([train_df.iloc[:, 0].values]).T.astype('int32')


# %%
# Adapted from prof Mafalda Challenge Notebook
def data_sequence_generator(
        data,  # input data
        lookback,  # how many timesteps back the input should go
        lookahead,  # how many timesteps in the future the target should be
        min_index,  # min index in the data array that delimits which timesteps to draw from
        max_index,  # max index in the data array that delimits which timesteps to draw from
        batch_size,  # the number of samples per batch
        shuffle=False,  # whether to shuffle the samples or draw them in chronological order
        step=1  # the period, in timesteps, at which data is sampled
):
    if max_index is None:
        max_index = len(data) - lookahead - 1
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
            targets[j] = data[rows[j] + lookahead][1]
        yield samples, targets


# %%
# Auto encoder model for feature reduction
# Adapted from https://blog.keras.io/building-autoencoders-in-keras.html
def make_autoencoder(
        lr=0.001,
        enc_dim=5
):
    ae0 = Input(shape=(544,), name='FeaturesInput')
    encode = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1), name='AE_feature_reduction')(ae0)
    decode = Dense(544, activation='relu', name='AE_3')(encode)

    autoencoder = Model(inputs=ae0, outputs=decode)
    encoder = Model(inputs=ae0, outputs=encode)

    autoencoder.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    return autoencoder, encoder


# adapted from:
# https://datascience.stackexchange.com/questions/26103/merging-two-different-models-in-keras
def make_model(
        lr=0.001,
        enc_dim=5
):
    # Auto-encoder for the features (categorical and quantitative)
    ae1 = Input(shape=(,544), name='FeaturesInput')
    ae2 = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1), name='AE_feature_reduction')(ae1)
    ae3 = Dense(544, activation='relu', name='AE_3')(ae2)

    # Recurrent layers
    rnn0 = Input(shape=(,,2), name='TimeSeriesInput')
    rnn1 = Concatenate(axis=1, name='ConcatenateInputs')([rnn0, ae2])
    rnn2 = SimpleRNN(10, activation='relu', name='RecurrentNN')(rnn1)
    rnn3 = Dropout(0.2, name='DropOut_layer')(rnn2)
    rnn4 = BatchNormalization(name='Batch_Normalization')(rnn3)
    rnn5 = Dense(1, activation='sigmoid', name='TimeSeries_output_layer')(rnn4)

    # Merge
    model = Model(inputs=[ae1, rnn0], outputs=[ae3, rnn5])

    # Compile
    model.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    return model


def plot_nn_metrics(nn_history, parameters=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if parameters is None:
        p_metrics = ['loss', 'mse']
    else:
        p_metrics = parameters
    nr_param = len(p_metrics)
    if round(nr_param / 2) == 1:
        matrix = (2, 1)
    else:
        matrix = (round(nr_param / 2), round(nr_param / 2))
    plt.figure(figsize=(12, 10))
    for n, metric in enumerate(p_metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(matrix[0], matrix[1], n + 1)
        plt.plot(nn_history.epoch, nn_history.history[metric], color=colors[0], label='Train')
        plt.plot(nn_history.epoch, nn_history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)

        plt.legend()
    plt.show()


# %%
autoenc, feat_encoder = make_autoencoder(0.001)
autoenc.summary()
feat_encoder.summary()
# %%
x_train, x_test = train_test_split(train_feat, test_size=0.25, shuffle=True, random_state=1)
x_train = minmax.fit_transform(x_train)
x_test = minmax.fit_transform(x_test)

history = autoenc.fit(
    x=x_train, y=x_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_test, x_test),
    callbacks=[EarlyStopping(
        patience=5,
        restore_best_weights=True
    )]
)
plot_nn_metrics(history)
# %%
pred = autoenc.predict(minmax.fit_transform(train_feat), batch_size=64)
pred_enc = feat_encoder.predict(minmax.fit_transform(train_feat), batch_size=64)

# %%
# Creates and TSNE model and plots it
# Adapted from https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
def tsne_plot(data, labels, annotate=False):
    products = []
    vector = []

    for l, v in zip(labels, data):
        products.append(l)
        vector.append(v)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=1, n_jobs=-1)
    new_values = tsne_model.fit_transform(vector)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        if annotate is True:
            plt.annotate(
                labels[i],
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom'
            )

    plt.show()

#%%
tsne_plot(pred_enc, train_feat_labels)

#%%
model = make_model()
model.summary()
