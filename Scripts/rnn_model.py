import numpy as np
import data_preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Input, \
    BatchNormalization, Concatenate, Reshape, TimeDistributed, Permute
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import he_uniform, he_normal
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.backend import transpose, squeeze
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

# %% Loading Product Data
train_prod_feat = np.array(data_preprocessing.product_train_df_zVersion.values).astype('int32')
test_prod_feat = np.array(data_preprocessing.product_test_df_zVersion.values).astype('int32')

products_shape = (train_prod_feat.shape[-1],)

# %% Loading of Time Series (each column corresponds to the index of above product data
train_ts = np.array(data_preprocessing.pts_train_df_zVersion.values).astype('int32')

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
    ae0 = Input(shape=products_shape, name='FeaturesInput')
    encode = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1), name='AE_feature_reduction')(ae0)
    decode = Dense(products_shape[0], activation='relu', name='AE_3')(encode)

    # inspired by https://www.frontiersin.org/articles/10.3389/fgene.2018.00585/full
    # clustering layers (will work with the help of OPTICS)
    # we want to find the probability of one product to be in 1 of total found clusters
    opt = OPTICS()
    opt.fit(minmax.fit_transform(data))
    clusters = len(np.unique(opt.labels_))
    print('Optimal number of cluster:', clusters)
    prob0 = Dense(enc_dim // 2, activation='relu', kernel_initializer=he_normal(1))(encode)
    prob1 = BatchNormalization()(prob0)
    prob = Dense(clusters, activation='softmax', name='Probability_Product')(prob1)

    autoencoder_ = Model(inputs=ae0, outputs=decode)
    encoder_ = Model(inputs=ae0, outputs=encode)
    p_prob = Model(inputs=ae0, outputs=prob)

    autoencoder_.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    return autoencoder_, encoder_, p_prob, opt


# %% Function to plot the neural network metrics
def plot_nn_metrics(nn_history, title=None, parameters=None):
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
        if title is not None:
            plt.title(title)

        plt.legend()
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
    autoenc, feat_encoder, _, _ = make_autoencoder(train_prod_feat, 0.001, i)

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
autoenc, feat_encoder, probab, _ = make_autoencoder(train_prod_feat, 0.001, 200)
autoenc.summary()
print('---------------------------------------------------------------------------------------------------------------')
feat_encoder.summary()

# %% Training of Auto-encoder
EPOCHS = 100
x_train, x_val = train_test_split(
    train_prod_feat,
    test_size=0.25,
    shuffle=True,
    random_state=1
)
x_train = minmax.fit_transform(x_train)
x_val = minmax.fit_transform(x_val)

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

#%%
tsne_plot(feat_encoder.predict(minmax.fit_transform(train_prod_feat)), range(0, train_prod_feat.shape[0]+1), True)

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
            rows = np.arange(
                _i,
                min(_i + batch_size, max_index)
            )
            _i += len(rows)

        samples = np.zeros(
            (
                len(rows),
                lookback // step,
                data.shape[-1]
            )
        )

        targets = np.zeros(
            (len(rows),)
        )

        for j, row in enumerate(rows):
            indices = range(
                rows[j] - lookback,
                rows[j],
                step
            )
            samples[j] = data[indices]
            targets[j] = data[rows[j] + target][1]

        # with yield the function returns a generator, instead of an array of arrays,
        # that will be feed to an fit_generator method of our NN model
        yield samples, targets


# %%
lookback = 6
delay = 6
batch_size = 12 * 3

train_gen = data_sequence_generator(
    data=minmax.fit_transform(train_ts),
    lookback=lookback,
    target=delay,
    batch_size=batch_size,
    max_index=50
)

val_gen = data_sequence_generator(
    data=minmax.fit_transform(train_ts),
    lookback=lookback,
    target=delay,
    batch_size=batch_size,
    min_index=51
)

#%%
def make_rnn(
        n_months=12,
        lr=0.001
):
    # Simple RNN layers
    seq_input = Input(shape=(None, train_ts.shape[-1]))  # Shape: (timesteps, data dimensions)
    # the number of units is the number of sequential months to predict
    rnn0 = SimpleRNN(n_months, activation='relu', return_sequences=True)(seq_input)
    out = TimeDistributed(Dense(1))(rnn0)

    model_ = Model(inputs=seq_input, outputs=out)
    model_.compile(
        optimizer=Adam(learning_rate=lr),  # ADAM is more appropriate for high dimensional data
        loss='mae',
        metrics=['mse']
    )

    return model_

#%%
model_rnn = make_rnn()
model_rnn.summary()

#%%
rnn_history = model_rnn.fit_generator(
    generator=train_gen,
    steps_per_epoch=100,
    epochs=25,
    callbacks=[EarlyStopping(
        patience=5,
        restore_best_weights=True
    )],
    validation_data=val_gen,
    validation_steps=100
)

plot_nn_metrics(rnn_history)

#%% Model conjugating Autoencoder and Simple RNN
def make_ae_rnn(
        product_shape,
        lr=0.001,
        enc_dim=200
):
    # Auto encoder layers
    ae0 = Input(shape=product_shape, name='FeaturesInput')
    encode = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1), name='AE_feature_reduction')(ae0)
    decode = Dense(product_shape[0], activation='relu', name='AE_3')(encode)
    perm = Permute((2, 1))(encode)

    # Simple RNN layers
    # inspired by https://dlpm2016.fbk.eu/docs/esteban_combining.pdf,
    # https://stackoverflow.com/questions/52474403/keras-time-series-suggestion-for-including-static-and-dynamic-variables-in-lstm,
    # https://blog.nirida.ai/predicting-e-commerce-consumer-behavior-using-recurrent-neural-networks-36e37f1aed22
    # https://www.affineanalytics.com/blog/new-product-forecasting-using-deep-learning-a-unique-way/
    # https://lilianweng.github.io/lil-log/2017/07/22/predict-stock-prices-using-RNN-part-2.html
    seq_input = Input(shape=(length, train_ts.shape[-1]))  # Shape: (timesteps, data dimensions)
    concat0 = Concatenate(axis=1)([seq_input, perm])
    rnn0 = SimpleRNN(train_ts.shape[-1], activation='relu', return_sequences=True)(concat0)
    # con1 = Reshape((-1, n_months))(con0)
    concat = Concatenate()([rnn0, con0])
    out = TimeDistributed(Dense(1))(concat)

    autoencoder_ = Model(inputs=ae0, outputs=decode)
    encoder_ = Model(inputs=ae0, outputs=encode)
    # model_rnn_ = Model(inputs=seq_input, outputs=out)
    model_full_ = Model(inputs=[ae0, seq_input], outputs=[out])

    # model_rnn_.compile(
    #     optimizer=Adam(learning_rate=lr),
    #     loss='mae',
    #     metrics=['mse']
    # )

    autoencoder_.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    model_full_.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    return autoencoder_, encoder_, model_full_

#%%
ae, enc, model_full = make_ae_rnn(
    product_shape=train_prod_feat.shape
)

# %%
model_full.summary()
plot_model(model_full, show_shapes=True, expand_nested=True)
