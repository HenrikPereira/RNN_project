import numpy as np
import data_preprocessing
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Input, \
    BatchNormalization, TimeDistributed, LSTM, GRU, LeakyReLU
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.initializers import he_uniform, he_normal
from tensorflow.keras.activations import relu, tanh, selu, linear
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


# %%
class config:
    """
    Configuration class for all primary objects, variables, and functions
    """

    def __init__(self, timeseries, product_train, product_test):
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.train_timeseries = np.array(timeseries.values).astype('int32')
        self.train_products = np.array(product_train.values).astype('int32')
        self.test_products = np.array(product_test.values).astype('int32')
        self.total_train_products = len(self.train_products)
        self.products_features = self.train_products.shape[1]
        self.total_timesteps = len(self.train_timeseries)

        # timeseries
        self.ts_batch_size = 50
        self.ts_lookback = 2  # same as length
        self.ts_delay = 1
        self.ts_split_index = 80
        self.ts_step = 1
        self.ts_steps_per_epoch = 20

    def split_products_train_val(self):
        val_split = 0.25
        random_state = 1
        x_train, x_val = train_test_split(
            self.train_products,
            test_size=val_split,
            shuffle=True,
            random_state=random_state
        )
        minmax = MinMaxScaler()
        minmax.fit(x_train)
        x_train = minmax.transform(x_train)
        x_val = minmax.transform(x_val)

        return x_train, x_val

    def plot_nn_metrics(self, nn_history, title=None, parameters=None):
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
            plt.subplot(matrix[v], matrix[h], n + 1)
            plt.plot(nn_history.epoch, nn_history.history[metric],
                     color=self.colors[0], label='Train')
            plt.plot(nn_history.epoch, nn_history.history['val_' + metric],
                     color=self.colors[1], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.legend()
            if h + 1 == matrix[1]:
                v += 1
                h = 0
            else:
                h += 1

        if title is not None:
            plt.title(title)

        plt.show()

    def data_sequence_generator(self, lookback, target, batch_size, aux_data=None,
                                min_index=0, max_index=None, shuffle=False, step=1):
        """

        Args:
            lookback: how many timesteps back the input should go
            target: how many timesteps in the future the target should be
            batch_size: the number of samples per batch
            aux_data: data to be used as header for each batch of samples and targets
            min_index: min index in the data array that delimits which timesteps to draw from
            max_index: max index in the data array that delimits which timesteps to draw from
            shuffle: whether to shuffle the samples or draw them in chronological order
            step: the period, in timesteps, at which data is sampled

        Returns:
            A data generator
        """
        if max_index is None:
            max_index = len(self.train_timeseries) - target - 1

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

            if aux_data is None:
                # both samples and targets are of equal dimensions because they have same number of features
                samples = np.zeros((len(rows), lookback // step, self.train_timeseries.shape[-1]))
                targets = np.zeros((len(rows), lookback // step, self.train_timeseries.shape[-1]))
            else:
                samples = np.zeros((len(rows), (lookback // step) + len(aux_data), self.train_timeseries.shape[-1]))
                targets = np.zeros((len(rows), (lookback // step) + len(aux_data), self.train_timeseries.shape[-1]))

            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                indices_target = range(rows[j] + target, (rows[j] + lookback) + target, step)
                if aux_data is not None:
                    samples[j] = np.append(aux_data, self.train_timeseries[indices], axis=0)
                    targets[j] = np.append(aux_data, self.train_timeseries[indices_target], axis=0)
                else:
                    samples[j] = self.train_timeseries[indices]
                    targets[j] = self.train_timeseries[indices_target]

            # with yield the function returns a generator, instead of an array of arrays,
            # that will be feed to an fit_generator method of our NN model
            yield samples, targets

    @staticmethod
    def get_prediction_from_generator(rnn_model, generator, n_steps, train_minmax):
        """
        Gets the prediction for the data generator used inside rnn model
        @param rnn_model: the rnn model to use method predict or predict generator
        @param generator: the data generator used.
        @param n_steps: number of iterations/timesteps to predict
        @param train_minmax: the MinMaxScaler instance fitted on the train data
        @return: array with predictions
        """
        predict_func = rnn_model.predict(generator, steps=n_steps)
        predicted = None
        for i in range(len(predict_func)):
            batch = train_minmax.inverse_transform(predict_func[i][1].reshape(1, -1))
            if predicted is None:
                predicted = batch
            else:
                predicted = np.append(predicted, batch, axis=0)


class nn:
    def __init__(self, learning_rate, product_features):
        self.lr = learning_rate
        self.prod_feat = product_features
        self.ae_metrics = ['mse', 'cosine_similarity']
        self.r_metrics = ['mse']
        self.loss = 'mae'
        self.EPOCHS = 500
        self.BATCH_SIZE = 64
        self.act_func_dict = [relu, LeakyReLU, tanh, selu, linear]
        self.optimizer_dict = ['adam', 'nadam', 'rmsprop']
        self.init_dict = [he_normal(1), he_uniform(1)]
        self.rnn_kind_dict = ['SimpleRNN', 'LSTM', 'GRU']

    @staticmethod
    def callbacks(save_checkpoint=False, model_name_to_file: str = 'model'):
        if model_name_to_file is None:
            model_name_to_file = 'model'

        early_stopping = EarlyStopping(
            patience=10,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            patience=5,
            cooldown=6,
            min_lr=0.00001
        )
        if save_checkpoint:
            checkpoint = ModelCheckpoint(
                filepath=model_name_to_file.join([r'./Logs/BestModels/', '.hdf5']),
                monitor='val_loss',
                save_best_only=True,
                save_freq='epoch'
            )
        else:
            checkpoint = None

        cb_set = [early_stopping, reduce_lr, checkpoint]

        return [i for i in cb_set if i is not None]

    # noinspection PyUnboundLocalVariable
    def make_ae(
            self,
            lr=0.001,  # learning rate
            n_hl_ae=1,  # number of hidden layers before and after encode layer [0,3]
            f_hl_ae=1,  # factor of neuron scaling on hidden layers besides encode layer
            enc_dim=200,  # target dimensions to reduce features
            initializer=0,  # kernel initializer for the dense layers
            act_func=0,  # activation function to use in dense layers
            opt=0,  # optimizer to use [adam, nadam, rmsprop]
            dropout=0,
            hidden_batch_norm=False,
            # bayes_nn=False
    ):
        # Auto encoder layers
        enc_l = Dense(enc_dim, activation=self.act_func_dict[act_func],
                      kernel_initializer=self.init_dict[initializer], name='Encoder')
        hl = Dense(enc_dim * f_hl_ae, activation=self.act_func_dict[act_func],
                   kernel_initializer=self.init_dict[initializer])
        drp = Dropout(dropout)
        bn = BatchNormalization()
        dec_l = Dense(self.prod_feat, activation=self.act_func_dict[act_func], name='Decoder')

        ae_in = Input(shape=(self.prod_feat,), name='FeaturesInput')
        if n_hl_ae == 0:
            encode = enc_l(ae_in)
            decode = dec_l(encode)
        elif n_hl_ae == 1:
            x = hl(ae_in)
            drp(ae_in)
            encode = enc_l(x)
            x = hl(encode)
            decode = dec_l(x)
        elif n_hl_ae == 2:
            x = hl(ae_in)
            x = drp(x)
            x = hl(x)
            if hidden_batch_norm:
                x = bn(x)
            encode = enc_l(x)
            x = hl(encode)
            x = hl(x)
            decode = dec_l(x)
        elif n_hl_ae == 3:
            x = hl(ae_in)
            x = drp(x)
            x = hl(x)
            x = hl(x)
            if hidden_batch_norm:
                x = bn(x)
            encode = enc_l(x)
            x = hl(encode)
            x = hl(x)
            x = hl(x)
            decode = dec_l(x)

        encoder = Model(inputs=ae_in, outputs=encode)
        autoencoder = Model(inputs=ae_in, outputs=decode)

        if self.optimizer_dict[opt] == 'adam':
            opt = Adam(learning_rate=lr)
        elif self.optimizer_dict[opt] == 'nadam':
            opt = Nadam(learning_rate=lr)
        elif self.optimizer_dict[opt] == 'rmsprop':
            opt = RMSprop(learning_rate=lr)

        autoencoder.compile(
            optimizer=self.optimizer_dict[opt],
            loss=self.loss,
            metrics=self.ae_metrics,
        )

        return autoencoder, encoder

    # noinspection PyUnboundLocalVariable
    def make_rnn(
            self,
            n_neurons,  # == length or lookback, to predict with length of output == length of timesteps inputted
            n_prod,  # number of products on the timeseries (features dimension)
            enc_ae_dim,  # number of encoded features of products (from autoencoder)
            step,  # number of steps on sliding window of the time series data generator
            lr=0.001,  # learning rate
            n_hl_r=0,  # number of recurrent hidden layers before main rnn layer [0, 3]
            f_hl_r=1,  # factor of neuron scaling on hidden layers besides main rnn layer
            rs_hl_r=False,  # sets return sequences value (True or False) in hidden layers
            act_func=2,  # activation function to use in rnn layers
            act_func_td=4,  # activation function to use in time TimeDistributed layers
            rnn_kind=0,  # type of recurrent layer to use [SimpleRNN, LSTM, GRU]
            opt=0,  # optimizer to use [adam, nadam, rmsprop]
            # bayes_nn=False
    ):
        # Simple RNN layers
        # the number of units is == the number of sequential months to predict on each batch
        if rnn_kind not in ['SimpleRNN', 'LSTM', 'GRU']:
            raise ValueError()

        rnn_l = SimpleRNN(n_neurons, activation=self.act_func_dict[act_func], return_sequences=True)
        rnn_hl = SimpleRNN(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r)
        if rnn_kind == 'LSTM':
            rnn_l = LSTM(n_neurons, activation=self.act_func_dict[act_func], return_sequences=True)
            rnn_hl = LSTM(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r)
        elif rnn_kind == 'GRU':
            rnn_l = GRU(n_neurons, activation=self.act_func_dict[act_func], return_sequences=True)
            rnn_hl = GRU(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r)

        seq_input = Input(shape=((n_neurons + enc_ae_dim) / step, n_prod))  # Shape: (timesteps, data dimensions)
        if n_hl_r == 0:
            r = rnn_l(seq_input)
        elif n_hl_r == 1:
            hl = rnn_hl(seq_input)
            r = rnn_l(hl)
        elif n_hl_r == 2:
            hl = rnn_hl(seq_input)
            hl = rnn_hl(hl)
            r = rnn_l(hl)
        elif n_hl_r == 3:
            hl = rnn_hl(seq_input)
            hl = rnn_hl(hl)
            hl = rnn_hl(hl)
            r = rnn_l(hl)
        out = TimeDistributed(Dense(n_prod, activation=act_func_td))(r)

        rnn = Model(inputs=seq_input, outputs=out)

        if opt == 'adam':
            opt = Adam(learning_rate=lr)
        elif opt == 'nadam':
            opt = Nadam(learning_rate=lr)
        elif opt == 'rmsprop':
            opt = RMSprop(learning_rate=lr)

        rnn.compile(
            optimizer=opt,
            loss=self.loss,
            metrics=self.r_metrics
        )

        return rnn

    def fit_ae(self, ae_model_obj, data, valid_data=None, model_name=None):
        """

        Args:
            ae_model_obj:
            data:
            valid_data:
            model_name:

        Returns:

        """
        if valid_data is not None:
            val = (valid_data, valid_data)
        else:
            val = None

        ae_history = ae_model_obj.fit(
            x=data, y=data,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=val,
            callbacks=self.callbacks(model_name_to_file=model_name),
            verbose=2
        )

        config.plot_nn_metrics(ae_history, parameters=['loss', 'cosine_similarity'])

    def fit_rnn(self, rnn_model_obj, data_gen, full_ts, ts_split_index, lookback,
                valid_gen=None, model_name=None, steps_p_epoch=20):
        if valid_gen is not None:
            val = valid_gen
        else:
            val = None

        val_steps = (len(full_ts) - ts_split_index - lookback)

        rnn_history = rnn_model_obj.fit_generator(
            generator=data_gen,
            steps_per_epoch=steps_p_epoch,
            epochs=self.EPOCHS,
            callbacks=self.callbacks(model_name_to_file=model_name),
            validation_data=val,
            validation_steps=val_steps,
            verbose=2
        )
        config.plot_nn_metrics(rnn_history)


class bayesian_opt:
    def __init__(self, init_config, type_nn: str, parameters: dict):
        self.parameters = parameters
        self.n_init_explore_point = 20
        self.n_bayesian_iterations = 20
        self.type_nn = type_nn
        assert type(init_config) == config
        self.c = init_config

    # TODO finish this part after asserting the rest of the methods
    def blackbox(self, data, valid_data):
        # Transform range of non discrete parameters into discrete values

        if self.type_nn == 'ae':
            model = nn.make_ae(

            )
            nn.fit_ae(
                ae_model_obj=model, data=data, valid_data=valid_data
            )
            scores = model.evaluate(
                x=data,
                y=valid_data,
                verbose=0
            )
        elif self.type_nn == 'rnn':
            model = nn.make_rnn(

            )
            nn.fit_rnn(
                rnn_model_obj=model, data_gen=data, full_ts=self.c.train_timeseries,
                ts_split_index=self.c.ts_split_index, lookback=self.c.ts_lookback,
                valid_gen=valid_data, steps_p_epoch=self.c.ts_steps_per_epoch
            )
            scores = model.evaluate_generator(
                generator=valid_data,
                verbose=0
            )

        return 1 / scores[
            0]  # this score has to be inverted in order to find the lowest value by maximizing the inverse

    def optimizer(self):
        tf.keras.backend.clear_session()

        def bb_partial(lr, n_hl_ae, f_hl_ae, enc_dim, initializer, act_func, opt, dropout, hidden_batch_norm,
                       n_neurons, n_prod, enc_ae_dim, step, n_hl_r, f_hl_r, rs_hl_r, act_func_td, rnn_kind):
            d = int(w)
            return self.blackbox(lr, n_hl_ae, f_hl_ae, enc_dim, initializer, act_func, opt, dropout, hidden_batch_norm,
                                 n_neurons, n_prod, enc_ae_dim, step, n_hl_r, f_hl_r, rs_hl_r, act_func_td, rnn_kind)

        b_opt = BayesianOptimization(
            f=bb_partial,
            pbounds=self.parameters,
            verbose=2,
            random_state=0
        )
        b_opt.maximize(
            init_points=self.n_init_explore_point,
            n_iter=self.n_bayesian_iterations,

        )
        print(b_opt.max)

        return b_opt


# %% instantiation of main configuration methods
i_config = config(
    timeseries=data_preprocessing.pts_train_df_zVersion,
    product_train=data_preprocessing.product_train_df_zVersion,
    product_test=data_preprocessing.product_test_df_zVersion
)

i_nn = nn(
    learning_rate=0.001,
    product_features=i_config.products_features
)

# %% instantiation of models

ae = i_nn.make_ae()
rnn = i_nn.make_rnn()

# %% preliminary fitting of models

i_nn.fit_ae(ae_model_obj=ae)
i_nn.fit_rnn(rnn_model_obj=rnn)



