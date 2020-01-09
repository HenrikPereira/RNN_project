import numpy as np
import data_preprocessing
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Input, \
    BatchNormalization, Concatenate, Reshape, TimeDistributed, Permute
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import he_uniform, he_normal
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

# %% Loading Product Data
train_prod_feat = np.array(data_preprocessing.product_train_df_zVersion.values).astype('int32')
test_prod_feat = np.array(data_preprocessing.product_test_df_zVersion.values).astype('int32')

products_shape = (train_prod_feat.shape[-1],)
prod_features = train_prod_feat.shape[-1]

# %% Loading of Time Series (each column corresponds to the index of above product data
train_ts = np.array(data_preprocessing.pts_train_df_zVersion.values).astype('int32')
total_timesteps = len(train_ts)
total_products = train_ts.shape[-1]

# %% MinMax Scaling
# this transformation can contaminate data along the whole dataset... use with caution...
minmax = MinMaxScaler()


# %% Auto encoder model for feature reduction
# Adapted from https://blog.keras.io/building-autoencoders-in-keras.html
def make_autoencoder(
        data,
        lr=0.001,
        enc_dim=100
):
    # Auto encoder layers
    ae0 = Input(shape=(products_shape[0],), name='FeaturesInput')
    encode = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1), name='AE_feature_reduction')(ae0)
    decode = Dense(products_shape[0], activation='relu', name='AE_3')(encode)

    # inspired by https://www.frontiersin.org/articles/10.3389/fgene.2018.00585/full
    # clustering layers (will work with the help of OPTICS)
    # we want to find the probability of one product to be in 1 of total found clusters
    # section on hold...
    # opt = OPTICS()
    # opt.fit(minmax.fit_transform(data))
    # clusters = len(np.unique(opt.labels_))
    # print('Optimal number of cluster:', clusters)
    # prob0 = Dense(enc_dim // 2, activation='relu', kernel_initializer=he_normal(1))(encode)
    # prob1 = BatchNormalization()(prob0)
    # prob = Dense(clusters, activation='softmax', name='Probability_Product')(prob1)

    autoencoder_ = Model(inputs=ae0, outputs=decode)
    encoder_ = Model(inputs=ae0, outputs=encode)
    # p_prob = Model(inputs=ae0, outputs=prob)

    autoencoder_.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    return autoencoder_, encoder_


# %% Function to plot the neural network metrics
def plot_nn_metrics(nn_history, title=None, parameters=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if parameters is None:
        p_metrics = ['loss', 'mse']
    else:
        p_metrics = parameters
    nr_param = len(p_metrics)
    if nr_param // 2 == 1:
        matrix = (2, 1)
    else:
        matrix = (round(nr_param / 2), round(nr_param / 2))
    plt.figure(figsize=(12, 10))
    v = 0
    h = 0
    for n, metric in enumerate(p_metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(matrix[i], matrix[1], n + 1)
        plt.plot(nn_history.epoch, nn_history.history[metric],
                 color=colors[0], label='Train')
        plt.plot(nn_history.epoch, nn_history.history['val_' + metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
        if h//matrix[1] == 1:
            v += 1
            h = 0
        else:
            h += 1


    if title is not None:
        plt.title(title)

    plt.show()


# %% Finding Optimal number of dimensions to reduce data
# Build log and plot to find optimal number of dimensions to compress the 544 products features
x_train, x_val = train_test_split(
    train_prod_feat,
    test_size=0.25,
    shuffle=True,
    random_state=1
)
x_train = minmax.fit_transform(x_train)
x_val = minmax.fit_transform(x_val)

min_mse = []
min_loss = []
dim = []
n_epochs = []
for i in np.arange(0, products_shape[0] - 4, 5):
    if i == 0:
        i = 1
    autoenc, feat_encoder = make_autoencoder(train_prod_feat, 0.001, i)

    history = autoenc.fit(
        x=x_train, y=x_train,
        batch_size=64,
        epochs=100,
        validation_data=(x_val, x_val),
        callbacks=[
            EarlyStopping(
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=2
    )

    dim.append(i)
    min_mse.append(history.history['val_mse'][-1])
    min_loss.append(history.history['val_loss'][-1])
    n_epochs.append(history.epoch[-1])

# Saving log to DataFrame and csv
ae_dim_log = pd.DataFrame(
    [i for i in zip(dim, min_mse, min_loss, n_epochs)],
    columns=['dimensions', 'last_mse', 'last_loss', 'n_epochs']
)
ae_dim_log.to_csv(r'./Logs/autoencoder_dimensions.csv')

plt.scatter(dim, min_mse, s=n_epochs, c='red', alpha=0.5)
plt.scatter(dim, min_loss, s=n_epochs, c='blue', alpha=0.5, marker='+')
plt.axhline(3e-2, c='gray', ls='dotted')
plt.axvline(200, c='gray', ls='dotted')
plt.show()

# After plotting the results, it can be seen that the encoding dimensions would suffice if between 100 and 200 to
# achieve least loss and MSE

# %% Initialization of Auto-encoder models
autoenc, feat_encoder = make_autoencoder(train_prod_feat, 0.001, 200)
autoenc.summary()
print('---------------------------------------------------------------------------------------------------------------')
feat_encoder.summary()

# %% Training of Auto-encoder
# split not necessary
EPOCHS = 100
x_train, x_val = train_test_split(
    train_prod_feat,
    test_size=0.25,
    shuffle=True,
    random_state=1
)

x_train = minmax.fit_transform(train_prod_feat)
x_val = minmax.fit_transform(train_prod_feat)

ae_history = autoenc.fit(
    x=x_train, y=x_train,
    batch_size=64,
    epochs=EPOCHS,
    validation_data=(x_val, x_val),
    callbacks=[EarlyStopping(
        patience=5,
        restore_best_weights=True
    )],
    verbose=0
)
plot_nn_metrics(ae_history)

# Save the trained weights of the autoencoder
autoenc.save_weights(r'./Logs/autoencoder.h5')

# Save the plot of the autoencoder
plot_model(autoenc, to_file=r'./Logs/autoencoder.png', expand_nested=True, show_shapes=True)


# %% t-SNE
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


# %%
tsne_plot(feat_encoder.predict(minmax.fit_transform(train_prod_feat)), range(0, train_prod_feat.shape[0] + 1), True)


# %% Data Sequence Generator
# Adapted from prof Mafalda Challenge Notebook
def data_sequence_generator(
        data,  # input data
        lookback,  # how many timesteps back the input should go
        target,  # how many timesteps in the future the target should be
        batch_size,  # the number of samples per batch
        min_index=0,  # min index in the data array that delimits which timesteps to draw from
        max_index=None,  # max index in the data array that delimits which timesteps to draw from
        shuffle=False,  # whether to shuffle the samples or draw them in chronological order
        step=1  # the period, in timesteps, at which data is sampled
):
    if max_index is None:
        max_index = len(data) - target - 1

    _i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(  # Return random integers from low (inclusive) to high (exclusive)
                low=min_index + lookback,
                high=max_index,
                size=batch_size
            )
        else:
            if _i + batch_size >= max_index:
                _i = min_index + lookback
            rows = np.arange(_i, min(_i + batch_size, max_index))
            _i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), lookback // step, data.shape[-1]))  # same dimensions as samples

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + target]  # removed [1]

        # with yield the function returns a generator, instead of an array of arrays,
        # that will be feed to an fit_generator method of our NN model
        yield samples, targets


# %%
batch_size = 50
length = 3
delay = 1
split_index = 80
step = 1
minmax.fit(train_ts)

train_gen = data_sequence_generator(
    data=minmax.transform(train_ts),
    lookback=length,
    target=delay,
    batch_size=batch_size,
    max_index=split_index,
    step=step
)

val_gen = data_sequence_generator(
    data=minmax.transform(train_ts),
    lookback=length,
    target=delay,
    batch_size=batch_size,
    min_index=split_index + 1,
    step=step
)


# train_tsgen = TimeseriesGenerator(
#     np.cbrt(train_ts),
#     np.cbrt(train_ts),
#     length=length,
#     sampling_rate=sample_rate,
#     batch_size=batch_size,
#     start_index=0,
#     end_index=86
# )
#
# val_tsgen = TimeseriesGenerator(
#     np.cbrt(train_ts),
#     np.cbrt(train_ts),
#     length=length,
#     sampling_rate=sample_rate,
#     batch_size=batch_size,
#     start_index=87
# )
#
# test_gen = TimeseriesGenerator(
#     np.cbrt(train_ts),
#     np.cbrt(train_ts),
#     length=length,
#     sampling_rate=sample_rate,
#     batch_size=batch_size
# )

# %%
def make_rnn(
        lr=0.001
):
    n_neurons = length  # we want the model to predict with length of output == to length of timesteps inputted

    # Simple RNN layers
    seq_input = Input(shape=(length / step, train_ts.shape[-1]))  # Shape: (timesteps, data dimensions)
    # the number of units is the number of sequential months to predict
    rnn0 = SimpleRNN(n_neurons, activation='tanh', return_sequences=True)(seq_input)
    out = TimeDistributed(Dense(train_ts.shape[-1], activation='relu'))(rnn0)

    model_ = Model(inputs=seq_input, outputs=out)
    model_.compile(
        optimizer=Adam(learning_rate=lr),  # ADAM is more appropriate for high dimensional data
        loss='mae',
        metrics=['mse']
    )

    return model_


# %%
model_rnn = make_rnn()
model_rnn.summary()

# %%
val_steps = (len(train_ts) - split_index + 1 - length)

rnn_history = model_rnn.fit_generator(
    generator=train_gen,
    steps_per_epoch=50,
    epochs=20,
    callbacks=[EarlyStopping(
        patience=5,
        restore_best_weights=True
    )],
    validation_data=val_gen,
    validation_steps=val_steps,
    verbose=2
)

plot_nn_metrics(rnn_history)

# %%
predict = model_rnn.predict(val_gen, steps=1)
predict = minmax.inverse_transform(predict[0])

# %%
# x_train = np.reshape(x_train, (-1, x_train.shape[0], x_train.shape[1]))
# x_val = np.reshape(x_val, (-1, x_val.shape[0], x_val.shape[1]))

# %% Model conjugating Autoencoder and Simple RNN
def make_ae_rnn(
        lr=0.001,
        enc_dim=200
):
    # Auto encoder layers
    # TODO allow the use of generator
    ae0 = Input(shape=(prod_features,), name='FeaturesInput')
    encode0 = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1))(ae0)
    encode = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1), name='AE_feature_reduction')(encode0)
    decode0 = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1))(encode)
    decode = Dense(prod_features, activation='relu', name='AE_3')(decode0)
    shape_re = Reshape((train_prod_feat.shape[0], enc_dim))(encode)
    perm = Permute((2, 1))(shape_re)
    # ae0 = Input(shape=(train_prod_feat.shape[0], prod_features,))
    # shape_re0 = Reshape((prod_features,))(ae0)
    # encode = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1))(shape_re0)
    # decode = Dense(prod_features, activation='relu', name='AE_3')(encode)
    # shape_re = Reshape((train_prod_feat.shape[0], enc_dim))(encode)
    # perm = Permute((2, 1))(shape_re)

    # Simple RNN layers
    # inspired by https://dlpm2016.fbk.eu/docs/esteban_combining.pdf,
    # https://stackoverflow.com/questions/52474403/keras-time-series-suggestion-for-including-static-and-dynamic-variables-in-lstm,
    # https://blog.nirida.ai/predicting-e-commerce-consumer-behavior-using-recurrent-neural-networks-36e37f1aed22
    # https://www.affineanalytics.com/blog/new-product-forecasting-using-deep-learning-a-unique-way/
    # https://lilianweng.github.io/lil-log/2017/07/22/predict-stock-prices-using-RNN-part-2.html
    n_neurons = length  # we want the model to predict with length of output == to length of timesteps inputted
    seq_input = Input(shape=(length / step, train_ts.shape[-1]))  # Shape: (timesteps, data dimensions)
    concat0 = Concatenate(axis=1)([perm, seq_input])
    # the number of units is the number of sequential months to predict
    rnn0 = SimpleRNN(n_neurons, activation='tanh', return_sequences=True)(concat0)
    out = TimeDistributed(Dense(train_ts.shape[-1]))(rnn0)

    encoder_ = Model(inputs=ae0, outputs=encode)
    autoencoder_ = Model(inputs=ae0, outputs=decode)
    autoencoder_.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse', 'cosine_similarity']
    )

    model_rnn_ = Model(inputs=[ae0, seq_input], outputs=out)
    model_rnn_.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    # new_prod_predictor_ = Model(inputs=ae0, outputs=out)

    model_full_ = Model(inputs=[ae0, seq_input], outputs=[out, decode])
    model_full_.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    return autoencoder_, encoder_, model_full_, model_rnn_


# %%
ae, enc, full, rnn = make_ae_rnn()

# %%
full.summary()
plot_model(full, to_file=r'./Logs/full_model.png', show_shapes=True, expand_nested=True)

# %%
ae.summary()
plot_model(ae, to_file=r'./Logs/autoencoder_model.png', show_shapes=True, expand_nested=True)

# %%
rnn.summary()
plot_model(rnn, to_file=r'./Logs/rnn_model.png', show_shapes=True, expand_nested=True)

# %%
x_train, x_val = train_test_split(
    train_prod_feat,
    test_size=0.20,
    shuffle=True,
    random_state=1
)
minmax.fit(x_train)
x_train = minmax.transform(x_train)
x_val = minmax.transform(x_val)

ae_history = ae.fit(
    x=x_train, y=x_train,
    batch_size=64,
    epochs=500,
    validation_data=(x_val, x_val),
    callbacks=[EarlyStopping(
        patience=20,
        restore_best_weights=True
    )],
    verbose=2
)
plot_nn_metrics(ae_history, parameters=['loss', 'mse', 'cosine_similarity'])

# prod_train_gen = data_sequence_generator(
#     x_train,
#     len(x_train),
#     0,
#     len(x_train)
# )
#
# ae_history = ae.fit_generator(
#     generator=prod_train_gen,
#     steps_per_epoch=100,
#     epochs=500,
#     validation_data=prod_train_gen,
#     validation_steps=1,
#     callbacks=[EarlyStopping(
#         patience=5,
#         restore_best_weights=True
#     )],
#     verbose=2
# )
# plot_nn_metrics(ae_history)

# %%
re_x_train = np.reshape(x_train, (-1, x_train.shape[0], x_train.shape[1]))
rnn.layers[0].trainable = False
rnn.layers[1].trainable = False
rnn.layers[2].trainable = False
rnn.layers[3].trainable = False
rnn.compile(
    optimizer=Adam(),
    loss='mae',
    metrics=['mse']
)
rnn.summary()

# rnn_history = rnn.fit(
#     x=[x_train, train_ts],
#     y=train_ts,
#     batch_size=64,
#     epochs=500,
#     validation_data=([x_train, train_ts], train_ts),
#     callbacks=[EarlyStopping(
#         patience=5,
#         restore_best_weights=True
#     )],
#     verbose=2
# )
# plot_nn_metrics(rnn_history)

rnn_history = rnn.fit_generator(
    generator=[re_x_train, train_gen],
    steps_per_epoch=100,
    epochs=500,
    callbacks=[EarlyStopping(
        patience=5,
        restore_best_weights=True
    )],
    validation_data=[re_x_train, val_gen],
    validation_steps=val_steps,
    verbose=2
)

# %%
full_history = full.fit_generator(
    generator=train_gen,
    steps_per_epoch=100,
    epochs=500,
    callbacks=[EarlyStopping(
        patience=5,
        restore_best_weights=True
    )],
    validation_data=val_gen,
    validation_steps=val_steps,
    verbose=2
)

plot_nn_metrics(full_history)
