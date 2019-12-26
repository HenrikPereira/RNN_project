import numpy as np
import data_preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, save_model, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Embedding, Input, \
    BatchNormalization, Concatenate, Flatten, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import he_uniform, he_normal
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Loading Product Data
train_prod_feat = np.array(data_preprocessing.product_train_df_zVersion.values).astype('int32')
test_prod_feat = np.array(data_preprocessing.product_test_df_zVersion.values).astype('int32')

products_shape = (train_prod_feat.shape[-1],)

# Loading of Time Series (each column corresponds to the index of above product data
train_ts = np.array(data_preprocessing.pts_train_df_zVersion.values).astype('int32')

# %%
# MinMax Scaling if necessary
# this transformation can contaminate data along the whole dataset... use with caution...
minmax = MinMaxScaler()

#%% ------> CODE STANDBY <------
# # Adapted from prof Mafalda Challenge Notebook
# def data_sequence_generator(
#         data,  # input data
#         lookback,  # how many timesteps back the input should go
#         lookahead,  # how many timesteps in the future the target should be
#         min_index,  # min index in the data array that delimits which timesteps to draw from
#         max_index,  # max index in the data array that delimits which timesteps to draw from
#         batch_size,  # the number of samples per batch
#         shuffle=False,  # whether to shuffle the samples or draw them in chronological order
#         step=1  # the period, in timesteps, at which data is sampled
# ):
#     if max_index is None:
#         max_index = len(data) - lookahead - 1
#     i = min_index + lookback
#     while 1:
#         if shuffle:
#             rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
#         else:
#             if i + batch_size >= max_index:
#                 i = min_index + lookback
#             rows = np.arange(i, min(i + batch_size, max_index))
#             i += len(rows)
#
#         samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
#
#         targets = np.zeros((len(rows),))
#
#         for j, row in enumerate(rows):
#             indices = range(rows[j] - lookback, rows[j], step)
#             samples[j] = data[indices]
#             targets[j] = data[rows[j] + lookahead][1]
#         yield samples, targets


# %%
# Auto encoder model for feature reduction
# Adapted from https://blog.keras.io/building-autoencoders-in-keras.html
def make_autoencoder(
        lr=0.001,
        enc_dim=100
):
    # Auto encoder layers
    ae0 = Input(shape=products_shape, name='FeaturesInput')
    encode = Dense(enc_dim, activation='relu', kernel_initializer=he_normal(1), name='AE_feature_reduction')(ae0)
    decode = Dense(products_shape[0], activation='relu', name='AE_3')(encode)

    # inspired by https://www.frontiersin.org/articles/10.3389/fgene.2018.00585/full
    # clustering layers (will work with the help of kmeans)
    # we want to find the probability of one product to be in 1 of 7 clusters
    prob0 = Dense(enc_dim//2, activation='relu', kernel_initializer=he_normal(1))(encode)
    prob1 = BatchNormalization()(prob0)
    prob = Dense(7, activation='softmax', name='Probability_Product')(encode)

    # # Recurrent layers
    # rnn0 = Input(shape=(1,), name='TimeSeriesInput')    # input of the outcome feature (1)
    # rnn1 = Concatenate(axis=1, name='ConcatenateInputs')([rnn0, ae2])   # Concatenate Outcome and product features
    # rnn2 = Reshape((10, 6))(rnn1)    # Reshape into (frame, (timesteps, features))
    # rnn3 = SimpleRNN(10, activation='relu', name='RecurrentNN')(rnn2)
    # rnn4 = Dropout(0.2, name='DropOut_layer')(rnn3)
    # rnn5 = BatchNormalization(name='Batch_Normalization')(rnn4)
    # rnn6 = Dense(1, activation='sigmoid', name='TimeSeries_output_layer')(rnn5)

    autoencoder = Model(inputs=ae0, outputs=decode)
    encoder = Model(inputs=ae0, outputs=encode)
    p_prob = Model(inputs=ae0, outputs=prob)
    # model_ = Model(inputs=[ae0, rnn0], outputs=[decode, rnn6])

    autoencoder.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )

    p_prob.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return autoencoder, encoder, p_prob


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

def plot_elbow_kmeans(df, max_ks=7):
    """
    :param df: Dataset to evaluate
    :param max_ks: maximum number of k nearest neighbours to be tested
    :return:
    """
    ks = np.arange(1, max_ks + 1)
    inertias = []

    for k in ks:
        # Create a KMeans instance with k clusters: model
        t_model = KMeans(n_clusters=k, n_jobs=-1, n_init=30, random_state=123, )
        # Fit model to samples
        t_model.fit(df)
        # Append the inertia to the list of inertias
        inertias.append(t_model.inertia_)

    # Calculate the distance between line k1 to kmax and ki cluster
    xi, yi = 1, inertias[0]
    xf, yf = max_ks, inertias[-1]

    distances = []
    for i, v in enumerate(inertias):
        x0 = i + 1
        y0 = v
        numerator = abs((yf - yi) * x0 - (xf - xi) * y0 + xf * yi - yf * xi)
        denominator = np.sqrt((yf - yi) ** 2 + (xf - xi) ** 2)
        distances.append(numerator / denominator)

    temp_df = pd.concat(
        [pd.Series(ks, name='ks'), pd.Series(inertias, name='Inertia'), pd.Series(distances, name='Distance')],
        axis=1).set_index('ks')

    xmax = temp_df['Distance'].idxmax()
    ymax = temp_df['Distance'].max()
    dmax = temp_df['Inertia'].loc[xmax]

    # Plot ks (x-axis) vs inertias (y-axis) using plt.plot().
    plt.figure(figsize=(10, 5))
    ax = sb.lineplot(data=temp_df.reset_index(), x='ks', y='Inertia')
    plt.axvline(xmax, c='r', ls=':')
    ax2 = ax.twinx()
    sb.lineplot(data=temp_df.reset_index(), x='ks', y='Distance', color='g', ax=ax2)

    # Annotations
    ax2.annotate('Max Distance', xy=(xmax, ymax), xytext=(xmax + 1, ymax),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('Elbow at k:{}'.format(xmax), xy=(xmax, dmax), xytext=(xmax + 1, dmax + 1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_title('K-means Inertia Graph (Elbow method)', fontsize=16)
    ax.set_xlabel('Nr of Clusters', fontsize=12)
    plt.xticks(ks)
    plt.tight_layout()

    plt.show()

    print('The best number of clusters is', xmax)

    return xmax


# %%
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
for i in np.arange(0, products_shape[0]-4, 5):
    if i == 0:
        i = 1
    autoenc, feat_encoder, _ = make_autoencoder(0.001, i)

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

#%%
autoenc, feat_encoder, product_probability = make_autoencoder(0.001, 200)
autoenc.summary()
print('---------------------------------------------------------------------------------------------------------------')
feat_encoder.summary()
print('---------------------------------------------------------------------------------------------------------------')
product_probability.summary()

#%%
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

# find optimal number of neighbours and
#plot_elbow_kmeans(feat_encoder.predict(x_train, batch_size=64), 25)

# Create labels of clusters to assign to products
k_means = KMeans(7, n_jobs=-1)
k_means.fit(feat_encoder.predict(x_train, batch_size=64))
y_train = to_categorical(k_means.labels_)
y_val = to_categorical(k_means.predict(feat_encoder.predict(x_val, batch_size=64)))

# the following layer has already been trained in the autoencoder, so we set it to trainable equal to false
# product_probability.layers[1].trainable = False

prob_history = product_probability.fit(
    x=x_train, y=y_train,
    batch_size=64,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[EarlyStopping(
        patience=10,
        restore_best_weights=True
    )],
    verbose=0
)
plot_nn_metrics(prob_history, parameters=['loss', 'accuracy'])

# Save the trained weights of the autoencoder
autoenc.save_weights(r'./Logs/autoencoder.h5')
product_probability.save_weights(r'./Logs/product_class.h5')

# Save the plot of the autoencoder
plot_model(autoenc, to_file=r'./Logs/autoencoder.png', expand_nested=True, show_shapes=True)
plot_model(product_probability, to_file=r'./Logs/product_class.png', expand_nested=True, show_shapes=True)

# %% ------> CODE STANDBY <------
# # Creates and TSNE model and plots it
# # Adapted from https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
# def tsne_plot(data, labels, annotate=False):
#     products = []
#     vector = []
#
#     for l, v in zip(labels, data):
#         products.append(l)
#         vector.append(v)
#
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=1, n_jobs=-1)
#     new_values = tsne_model.fit_transform(vector)
#
#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
#
#     plt.figure(figsize=(16, 16))
#     for i in range(len(x)):
#         plt.scatter(x[i], y[i])
#         if annotate is True:
#             plt.annotate(
#                 labels[i],
#                 xy=(x[i], y[i]),
#                 xytext=(5, 2),
#                 textcoords='offset points',
#                 ha='right',
#                 va='bottom'
#             )
#
#     plt.show()
#
# #%%
# tsne_plot(pred_enc, train_prod_labels, True)
#
# #%%
# tsne_plot(minmax.fit_transform(train_prod_feat), train_prod_labels, True)

#%%
pred = product_probability.predict(test_prod_feat, batch_size=64)
pred1 = product_probability.predict(train_prod_feat, batch_size=64)

