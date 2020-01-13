# Import of essential libraries
import pickle
import numpy as np
import data_preprocessing
from math import ceil
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
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

# Classes for the easy implementation of our models
class config:
    """
    Configuration class for all primary objects, variables, and functions
    """
    allowed_keys = {'ts_batch_size', 'ts_lookback', 'ts_delay', 'ts_split_index', 'ts_step', 'ts_steps_per_epoch'}

    def __init__(self, timeseries, product_train, product_test, **kwargs):
        """

        Args:
            timeseries:
            product_train:
            product_test:
            **kwargs:
        """
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
        self.ts_split_index = 0
        self.ts_step = 1
        self.ts_steps_per_epoch = 20
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in config.allowed_keys)

        # product splits
        self.x_train = None
        self.x_train_minmax = None
        self.x_val = None

    def split_products_train_val(self, val_split=0.25, random_state=1):
        """

        Args:
            val_split:
            random_state:

        Returns:

        """
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

        self.x_train = x_train
        self.x_train_minmax = minmax
        self.x_val = x_val
        print('Done...')

    def data_sequence_generator(self, aux_data=None, min_index=0, max_index=None, shuffle=False):
        """

        Args:
            aux_data: data to be used as header for each batch of samples and targets
            min_index: min index in the data array that delimits which timesteps to draw from
            max_index: max index in the data array that delimits which timesteps to draw from
            shuffle: whether to shuffle the samples or draw them in chronological order

        Returns:
            A data generator
        """
        if max_index is None:
            max_index = len(self.train_timeseries) - self.ts_delay - 1

        _i = min_index + self.ts_lookback
        while 1:
            if shuffle:
                rows = np.random.randint(  # Return random integers from low (inclusive) to high (exclusive)
                    low=min_index + self.ts_lookback,
                    high=max_index,
                    size=self.ts_batch_size
                )
            else:
                if _i + self.ts_batch_size >= max_index:
                    _i = min_index + self.ts_lookback
                rows = np.arange(_i, min(_i + self.ts_batch_size, max_index))
                _i += len(rows)

            if aux_data is None:
                # both samples and targets are of equal dimensions because they have same number of features
                samples = np.zeros((len(rows), self.ts_lookback // self.ts_step, self.train_timeseries.shape[-1]))
                targets = np.zeros((len(rows), 1, self.train_timeseries.shape[-1]))
            else:
                samples = np.zeros(
                    (len(rows), (self.ts_lookback // self.ts_step) + len(aux_data), self.train_timeseries.shape[-1]))
                targets = np.zeros(
                    (len(rows), 1, self.train_timeseries.shape[-1]))

            for j, row in enumerate(rows):
                indices = range(rows[j] - self.ts_lookback, rows[j], self.ts_step)
                indices_target = rows[j] + self.ts_delay
                if aux_data is not None:
                    samples[j] = np.append(aux_data, self.train_timeseries[indices], axis=0)
                    targets[j] = self.train_timeseries[indices_target]
                    # self.train_timeseries[indices_target].reshape(1, -1)
                else:
                    samples[j] = self.train_timeseries[indices]
                    targets[j] = self.train_timeseries[indices_target]

            # with yield the function returns a generator, instead of an array of arrays,
            # that will be feed to an fit_generator method of our NN model
            yield samples, targets

    @staticmethod
    def get_prediction_from_generator(rnn_model, generator, n_steps, train_minmax):
        """

        Args:
            rnn_model: the rnn model to use method predict or predict generator
            generator: the data generator to be used to predict new values.
            n_steps: number of iterations/timesteps to predict
            train_minmax: the MinMaxScaler instance fitted on the train data

        Returns:
            array with predictions
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
    """

    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def __init__(self, learning_rate, product_features):
        """

        Args:
            learning_rate:
            product_features:
        """
        self.lr = learning_rate
        self.prod_feat = product_features
        self.ae_metrics = ['mse', 'cosine_similarity']
        self.r_metrics = ['mse']
        self.loss = 'mae'
        self.EPOCHS = 500
        self.BATCH_SIZE = 64
        self.act_func_dict = [relu, tanh, selu, linear, LeakyReLU]
        self.optimizer_dict = ['adam', 'nadam', 'rmsprop']
        self.init_dict = [he_normal(1), he_uniform(1)]
        self.rnn_kind_dict = ['SimpleRNN', 'LSTM', 'GRU']
        self.enc_dim = None

    @staticmethod
    def plot_nn_metrics(nn_history, title: str = None, parameters=None, save=False):
        """

        Args:
            nn_history:
            title:
            parameters:
            save:

        Returns:

        """
        if parameters is None:
            p_metrics = ['loss', 'mse']
        else:
            p_metrics = parameters
        nr_param = len(p_metrics)
        if nr_param == 2:
            matrix = (2, 1)
        else:
            matrix = (ceil(nr_param / 2), ceil(nr_param / 2))
        plt.figure(figsize=(12, 10))
        for n, metric in enumerate(p_metrics):
            if metric == tf.keras.metrics.RootMeanSquaredError:
                name = 'Rmse'
            else:
                name = metric.replace("_", " ").capitalize()
            plt.subplot(matrix[0], matrix[1], n + 1)
            plt.plot(nn_history.epoch, nn_history.history[metric],
                     color=nn.colors[0], label='Train')
            plt.plot(nn_history.epoch, nn_history.history['val_' + metric],
                     color=nn.colors[1], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.legend()

        if title is not None:
            plt.title(title)
            if save:
                plt.savefig(fname=r'./Logs/Figures/' + title.strip(' ') + 'png', transparent=True)

        plt.show()

    @staticmethod
    def callbacks(save_checkpoint=False, model_name_to_file: str = 'model'):
        """

        Args:
            save_checkpoint:
            model_name_to_file:

        Returns:

        """
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
    def make_ae(self, lr=0.001, n_hl_ae=1, f_hl_ae=1, enc_dim=200,
                initializer=0, act_func=0, opt=0, dropout=0, hidden_batch_norm=False):
        """

        Args:
            lr: learning rate
            n_hl_ae: number of hidden layers before and after encode layer [0,3]
            f_hl_ae: factor of neuron scaling on hidden layers besides encode layer
            enc_dim: target dimensions to reduce features
            initializer: kernel initializer for the dense layers
            act_func: activation function to use in dense layers
            opt: optimizer to use [adam, nadam, rmsprop]
            dropout:
            hidden_batch_norm:

        Returns:

        """
        self.enc_dim = enc_dim

        # Auto encoder layers
        enc_l = Dense(enc_dim, activation=self.act_func_dict[act_func],
                      kernel_initializer=self.init_dict[initializer], name='Encoder')
        hl_in = Dense(enc_dim * f_hl_ae, activation=self.act_func_dict[act_func],
                      kernel_initializer=self.init_dict[initializer], name='HiddenIn1st')
        hl_in2 = Dense(enc_dim * f_hl_ae, activation=self.act_func_dict[act_func],
                       kernel_initializer=self.init_dict[initializer], name='HiddenIn2nd')
        hl_in3 = Dense(enc_dim * f_hl_ae, activation=self.act_func_dict[act_func],
                       kernel_initializer=self.init_dict[initializer], name='HiddenIn3rd')
        hl_out = Dense(enc_dim * f_hl_ae, activation=self.act_func_dict[act_func],
                       kernel_initializer=self.init_dict[initializer], name='HiddenOut1st')
        hl_out2 = Dense(enc_dim * f_hl_ae, activation=self.act_func_dict[act_func],
                        kernel_initializer=self.init_dict[initializer], name='HiddenOut2nd')
        hl_out3 = Dense(enc_dim * f_hl_ae, activation=self.act_func_dict[act_func],
                        kernel_initializer=self.init_dict[initializer], name='HiddenOut3rd')
        drp = Dropout(dropout)
        bn = BatchNormalization()
        dec_l = Dense(self.prod_feat, activation=self.act_func_dict[act_func], name='Decoder')

        ae_in = Input(shape=(self.prod_feat,), name='FeaturesInput')
        if n_hl_ae == 0:
            encode = enc_l(ae_in)
            decode = dec_l(encode)
        elif n_hl_ae == 1:
            x = hl_in(ae_in)
            x = drp(x)
            encode = enc_l(x)
            x = hl_out(encode)
            decode = dec_l(x)
        elif n_hl_ae == 2:
            x = hl_in(ae_in)
            x = drp(x)
            x = hl_in2(x)
            if hidden_batch_norm:
                x = bn(x)
            encode = enc_l(x)
            x = hl_out2(encode)
            x = hl_out(x)
            decode = dec_l(x)
        elif n_hl_ae == 3:
            x = hl_in(ae_in)
            x = drp(x)
            x = hl_in2(x)
            x = hl_in3(x)
            if hidden_batch_norm:
                x = bn(x)
            encode = enc_l(x)
            x = hl_out3(encode)
            x = hl_out2(x)
            x = hl_out(x)
            decode = dec_l(x)
        else:
            raise ValueError

        _encoder = Model(inputs=ae_in, outputs=encode)
        _autoencoder = Model(inputs=ae_in, outputs=decode)

        if self.optimizer_dict[opt] == 'adam':
            opt_ = Adam(learning_rate=lr)
        elif self.optimizer_dict[opt] == 'nadam':
            opt_ = Nadam(learning_rate=lr)
        elif self.optimizer_dict[opt] == 'rmsprop':
            opt_ = RMSprop(learning_rate=lr)
        else:
            raise ValueError
        _autoencoder.compile(
            optimizer=opt_,
            loss=self.loss,
            metrics=self.ae_metrics,
        )

        return _autoencoder, _encoder

    # noinspection PyUnboundLocalVariable
    def make_rnn(self, n_neurons, n_prod, step, enc_ae_dim=0, lr=0.001,
                 n_hl_r=0, f_hl_r=1, rs_hl_r=False, act_func=2, act_func_td=3, rnn_kind=0, opt=0):
        """

        Args:
            n_neurons: length or lookback, to predict with length of output == length of timesteps inputted
            n_prod: number of products on the timeseries (features dimension)
            step: number of steps on sliding window of the time series data generator
            enc_ae_dim: number of encoded features of products (from autoencoder)
            lr: learning rate
            n_hl_r: number of recurrent hidden layers before main rnn layer [0, 3]
            f_hl_r: factor of neuron scaling on hidden layers besides main rnn layer
            rs_hl_r: sets return sequences value (True or False) in hidden layers
            act_func: activation function to use in rnn layers
            act_func_td: activation function to use in time TimeDistributed layers
            rnn_kind: type of recurrent layer to use [SimpleRNN, LSTM, GRU]
            opt: optimizer to use [adam, nadam, rmsprop]

        Returns:

        """
        # Simple RNN layers
        # the number of units is == the number of sequential months to predict on each batch
        if self.rnn_kind_dict[rnn_kind] not in ['SimpleRNN', 'LSTM', 'GRU']:
            raise ValueError()

        rnn_l = SimpleRNN(n_neurons, activation=self.act_func_dict[act_func], return_sequences=True,
                          name=self.rnn_kind_dict[rnn_kind])
        rnn_hl = SimpleRNN(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r,
                           name=self.rnn_kind_dict[rnn_kind] + 'hidden1st')
        rnn_hl2 = SimpleRNN(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r,
                            name=self.rnn_kind_dict[rnn_kind] + 'hidden2nd')
        if self.rnn_kind_dict[rnn_kind] == 'LSTM':
            rnn_l = LSTM(n_neurons, activation=self.act_func_dict[act_func], return_sequences=True,
                         name=self.rnn_kind_dict[rnn_kind])
            rnn_hl = LSTM(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r,
                          name=self.rnn_kind_dict[rnn_kind] + 'hidden1st')
            rnn_hl2 = LSTM(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r,
                           name=self.rnn_kind_dict[rnn_kind] + 'hidden2nd')
        elif self.rnn_kind_dict[rnn_kind] == 'GRU':
            rnn_l = GRU(n_neurons, activation=self.act_func_dict[act_func], return_sequences=True,
                        name=self.rnn_kind_dict[rnn_kind])
            rnn_hl = GRU(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r,
                         name=self.rnn_kind_dict[rnn_kind] + 'hidden1st')
            rnn_hl2 = GRU(n_neurons * f_hl_r, activation=self.act_func_dict[act_func], return_sequences=rs_hl_r,
                          name=self.rnn_kind_dict[rnn_kind] + 'hidden2nd')

        seq_input = Input(shape=((n_neurons + enc_ae_dim) / step, n_prod))  # Shape: (timesteps, data dimensions)
        if n_hl_r == 0:
            r = rnn_l(seq_input)
        elif n_hl_r == 1:
            hl = rnn_hl(seq_input)
            r = rnn_l(hl)
        elif n_hl_r == 2:
            hl = rnn_hl(seq_input)
            hl = rnn_hl2(hl)
            r = rnn_l(hl)
        elif n_hl_r == 3:
            hl = rnn_hl(seq_input)
            hl = rnn_hl2(hl)
            hl = rnn_hl2(hl)
            r = rnn_l(hl)
        out = TimeDistributed(Dense(n_prod, activation=self.act_func_dict[act_func_td]))(r)

        _rnn = Model(inputs=seq_input, outputs=out)

        if self.optimizer_dict[opt] == 'adam':
            opt_ = Adam(learning_rate=lr)
        elif self.optimizer_dict[opt] == 'nadam':
            opt_ = Nadam(learning_rate=lr)
        elif self.optimizer_dict[opt] == 'rmsprop':
            opt_ = RMSprop(learning_rate=lr)
        else:
            raise ValueError
        _rnn.compile(
            optimizer=opt_,
            loss=self.loss,
            metrics=self.r_metrics
        )

        return _rnn

    def fit_ae(self, ae_model_obj, data, valid_data=None, model_name=None, **kwargs):
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

        if 'epochs' in kwargs.keys():
            _epochs = kwargs['epochs']
        else:
            _epochs = self.EPOCHS

        if 'verbose' in kwargs.keys():
            _verbose = kwargs['verbose']
        else:
            _verbose = 2

        ae_history = ae_model_obj.fit(
            x=data, y=data,
            batch_size=self.BATCH_SIZE,
            epochs=_epochs,
            validation_data=val,
            callbacks=self.callbacks(model_name_to_file=model_name),
            verbose=_verbose
        )

        nn.plot_nn_metrics(nn_history=ae_history, parameters=['loss', 'mse', 'cosine_similarity'])

        return ae_history

    def fit_rnn(self, rnn_model_obj, data_gen, full_ts, lookback, ts_split_index=0,
                valid_gen=None, model_name=None, steps_p_epoch=20, **kwargs):
        """

        Args:
            rnn_model_obj:
            data_gen:
            full_ts:
            lookback:
            ts_split_index:
            valid_gen:
            model_name:
            steps_p_epoch:
            **kwargs:

        Returns:

        """
        if valid_gen is not None:
            val = valid_gen
        else:
            val = None

        if 'epochs' in kwargs.keys():
            _epochs = kwargs['epochs']
        else:
            _epochs = self.EPOCHS

        if 'verbose' in kwargs.keys():
            _verbose = kwargs['verbose']
        else:
            _verbose = 2

        val_steps = (len(full_ts) - ts_split_index - lookback)

        rnn_history = rnn_model_obj.fit_generator(
            generator=data_gen,
            steps_per_epoch=steps_p_epoch,
            epochs=_epochs,
            callbacks=self.callbacks(model_name_to_file=model_name),
            validation_data=val,
            validation_steps=val_steps,
            verbose=_verbose
        )
        nn.plot_nn_metrics(nn_history=rnn_history, parameters=['loss', 'mse'])

        return rnn_history


class bayesian_opt:
    """

    """
    allowed_continuous_keys = {'lr', 'dropout'}
    allowed_discrete_keys = {
        'n_hl_ae', 'f_hl_ae', 'enc_dim', 'initializer', 'act_func', 'opt',
        'hidden_batch_norm', 'n_neurons', 'enc_ae_dim', 'n_hl_r', 'f_hl_r',
        'rs_hl_r', 'act_func_td', 'rnn_kind'}

    def __init__(
            self, init_config: config, nn_config: nn, type_nn: str,
            parameters: dict, data_gen=None, type_rnn: int = None, enc_dim: int = None):
        """

        Args:
            init_config:
            type_nn:
            parameters:
            **kwargs (object):
        """
        self.parameters = parameters
        self.n_init_explore_point = 20
        self.n_bayesian_iterations = 10
        self.type_nn = type_nn
        self.c = init_config
        self.data_gen = data_gen
        self.nn_c = nn_config
        self.model = None

        # kwargs for the blackbox
        # continuous
        self.lr = None
        self.dropout = None

        # discrete
        self.n_hl_ae = None
        self.f_hl_ae = None
        self.enc_dim = None
        self.initializer = None
        self.act_func = None
        self.opt = None
        self.hidden_batch_norm = None
        self.n_neurons = None
        self.enc_ae_dim = enc_dim
        self.n_hl_r = None
        self.f_hl_r = None
        self.rs_hl_r = True
        self.act_func_td = None
        self.rnn_kind = type_rnn

    def blackbox(self):
        if self.type_nn == 'ae':
            self.model = self.nn_c.make_ae(
                lr=self.lr,
                n_hl_ae=self.n_hl_ae,
                f_hl_ae=self.f_hl_ae,
                enc_dim=self.enc_dim,
                initializer=self.initializer,
                act_func=self.act_func,
                opt=self.opt,
                dropout=self.dropout,
                hidden_batch_norm=self.hidden_batch_norm
            )
            self.nn_c.fit_ae(
                ae_model_obj=self.model[0], data=self.c.x_train, valid_data=self.c.x_val,
                verbose=0
            )
            scores = self.model[0].evaluate(
                x=self.c.x_train,
                y=self.c.x_train,
                verbose=0
            )
        elif self.type_nn == 'rnn':
            self.model = self.nn_c.make_rnn(
                n_neurons=self.c.ts_lookback, n_prod=self.c.total_train_products, step=self.c.ts_step,
                enc_ae_dim=self.enc_ae_dim,
                lr=self.lr,
                n_hl_r=self.n_hl_r,
                f_hl_r=self.f_hl_r,
                rs_hl_r=self.rs_hl_r,
                act_func=self.act_func,
                act_func_td=self.act_func_td,
                rnn_kind=self.rnn_kind,
                opt=self.opt
            )
            self.nn_c.fit_rnn(
                rnn_model_obj=self.model, data_gen=self.data_gen, full_ts=self.c.train_timeseries,
                ts_split_index=0, lookback=self.c.ts_lookback,
                valid_gen=self.data_gen, steps_p_epoch=self.c.ts_steps_per_epoch, epochs=20,
                verbose=0
            )
            scores = self.model.evaluate_generator(
                generator=self.data_gen,
                steps=5,
                verbose=0
            )

        # this score has to be inverted in order to find the lowest value by maximizing the inverse
        return 1 / scores[0]

    def bb_partial(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in bayesian_opt.allowed_continuous_keys)
        self.__dict__.update((k, int(v)) for k, v in kwargs.items() if k in bayesian_opt.allowed_discrete_keys)

        return self.blackbox()

    def optimizer(self, log_json=False, **kwargs):
        allow = ['n_init_explore_point', 'n_bayesian_iterations']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allow)

        b_opt = BayesianOptimization(
            f=self.bb_partial,
            pbounds=self.parameters,
            verbose=2,
            random_state=0
        )

        if log_json:
            logger = JSONLogger(path=r'./Logs/' + self.type_nn + '_' + str(self.rnn_kind) + '.json')
            b_opt.subscribe(Events.OPTMIZATION_STEP, logger)

        b_opt.maximize(
            init_points=self.n_init_explore_point,
            n_iter=self.n_bayesian_iterations,
            acq="poi",  # Acquisition Function "Probability of Improvement" --> Prefer exploration (with xi=0.1)
            xi=1e-1
        )
        print('best parameters:', b_opt.max)

        return b_opt

    def param_decode(self, optimizer_max_params):
        keys = optimizer_max_params.keys()
        values = optimizer_max_params.values()
        _new = {}
        for k, v in zip(keys, values):
            if k in self.allowed_continuous_keys:
                _new[k] = v
            else:
                _new[k] = int(v)

        # for k in d_keys:
        #     self.nn_c.act_func_dict = [relu, tanh, selu, linear, LeakyReLU]
        #     self.nn_c.optimizer_dict = ['adam', 'nadam', 'rmsprop']
        #     self.nn_c.init_dict = [he_normal(1), he_uniform(1)]
        #     self.nn_c.rnn_kind_dict = ['SimpleRNN', 'LSTM', 'GRU']

        print(_new)
        return _new

# %% instantiation of main configuration methods
i_config = config(
    timeseries=data_preprocessing.pts_train_df_zVersion,
    product_train=data_preprocessing.product_train_df_zVersion,
    product_test=data_preprocessing.product_test_df_zVersion,
    ts_lookback=6
)
ts_gen_train = i_config.data_sequence_generator()
# TODO is validation of TS that really necessary? a product does not have multiple pre_release and releases...
# ts_gen_val = i_config.data_sequence_generator(min_index=i_config.ts_split_index + 1)

i_config.split_products_train_val()

i_nn = nn(learning_rate=0.001, product_features=i_config.products_features)

# instantiation of models
ae, enc = i_nn.make_ae(n_hl_ae=3)
rnn = i_nn.make_rnn(n_neurons=i_config.ts_lookback, n_prod=i_config.total_train_products, step=i_config.ts_step,
                    n_hl_r=1, rs_hl_r=True, rnn_kind=2)

# preliminary fitting of models
ae_hist = i_nn.fit_ae(ae_model_obj=ae, data=i_config.x_train, valid_data=i_config.x_val)

# this rnn configuration is just to check if all is working well
_ = i_nn.fit_rnn(
    rnn_model_obj=rnn, data_gen=ts_gen_train, full_ts=i_config.train_timeseries,
    ts_split_index=i_config.ts_split_index, lookback=i_config.ts_lookback,
    valid_gen=ts_gen_train, epochs=5
)

i_nn.plot_nn_metrics(ae_hist, title='Autoencoder3hl', save=True)

# instantiate new stacked timeseries generators
aux_enc = enc.predict(x=i_config.train_products, batch_size=i_nn.BATCH_SIZE).transpose()

stack_tsgen_t = i_config.data_sequence_generator(aux_data=aux_enc)

# make RNN using the stacked data generator
s_rnn = i_nn.make_rnn(
    n_neurons=i_config.ts_lookback, n_prod=i_config.total_train_products, step=i_config.ts_step,
    n_hl_r=1, rs_hl_r=True, rnn_kind=0, enc_ae_dim=i_nn.enc_dim)

# fitting of said RNN
s_rnn_h = i_nn.fit_rnn(
    rnn_model_obj=s_rnn, data_gen=stack_tsgen_t, full_ts=i_config.train_timeseries,
    lookback=i_config.ts_lookback, ts_split_index=0,
    valid_gen=stack_tsgen_t, epochs=10
)

# Models plots & figures & metrics
models = [ae, s_rnn]
histories = [ae_hist, s_rnn_h]
labels = ['auto_encoder', 'simple_rnn']

for i, m in zip(labels, models):
    plot_model(m, to_file=r'./Logs/Temp/' + i + 'model.png', expand_nested=True, show_shapes=True)


for i, h in zip(labels, histories):
    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    with open(r'./Logs/' + i, 'wb') as file_pi:
        pickle.dump(h.history, file_pi)
    while i == 0:
        i_nn.plot_nn_metrics(
            nn_history=h, parameters=['loss', 'mse', 'cosine_similarity'], title=i + 'metrics', save=True)
    else:
        i_nn.plot_nn_metrics(
            nn_history=h, parameters=['loss', 'mse'], title=i + 'metrics', save=True)

# Saving of histories to DataFrames
ae_hist_df = pd.DataFrame(ae_hist.history)
s_rnn_h_df = pd.DataFrame(s_rnn_h.history)

# clearing memory...
tf.keras.backend.clear_session()

# optimization of hyper-parameters for the auto-encoder NN
# TODO save best model hdf5
ae_params = {
    'lr': (0.1, 0.001),
    'n_hl_ae': (0, 3),
    'f_hl_ae': (1, 2),
    'enc_dim': (100, 300),
    'initializer': (0, 1),
    'act_func': (0, 3),
    'opt': (0, 2),
    'dropout': (0, 0.49),
    'hidden_batch_norm': (0, 1)
}
i_optimizer_ae = bayesian_opt(init_config=i_config, nn_config=i_nn, type_nn='ae', parameters=ae_params)
ae_opt = i_optimizer_ae.optimizer(n_init_explore_point=60, n_bayesian_iterations=60, log_json=True)

tf.keras.backend.clear_session()

# fitting of whole products using best hyper-parameters and generation of corresponding ts data generator
target_ae = i_optimizer_ae.param_decode(ae_opt.max['params'])

ae, enc = i_nn.make_ae(
    target_ae['lr'], target_ae['n_hl_ae'], target_ae['f_hl_ae'], target_ae['enc_dim'], target_ae['initializer'],
    target_ae['act_func'], target_ae['opt'], target_ae['dropout'], target_ae['hidden_batch_norm']
)

minmax_f = MinMaxScaler()
minmax_f.fit(i_config.train_products)
full_train = minmax_f.transform(i_config.train_products)

i_nn.fit_ae(
    ae_model_obj=ae, data=full_train, valid_data=full_train,
    model_name='final_ae', verbose=0)

aux_enc = enc.predict(i_config.train_products, i_nn.BATCH_SIZE)

stack_tsgen_t = i_config.data_sequence_generator(aux_data=aux_enc.transpose())

# %% optimization of hyper-parameters for the recurrent NN
# TODO save best model hdf5
s_rnn_params = {
    'lr': (0.1, 0.001),
    'n_hl_r': (0, 3),
    'f_hl_r': (1, 2),
    'act_func': (0, 3),
    'act_func_td': (1, 3),
    'opt': (0, 2)
}
lstm_rnn_params = {
    'lr': (0.1, 0.001),
    'n_hl_r': (0, 3),
    'f_hl_r': (1, 2),
    'act_func': (0, 3),
    'act_func_td': (1, 3),
    'opt': (0, 2)
}
gru_rnn_params = {
    'lr': (0.1, 0.001),
    'n_hl_r': (0, 3),
    'f_hl_r': (1, 2),
    'act_func': (0, 3),
    'act_func_td': (1, 3),
    'opt': (0, 2)
}
i_optimizer_srnn = bayesian_opt(
    init_config=i_config, nn_config=i_nn, type_nn='rnn', type_rnn=0,
    parameters=s_rnn_params, enc_dim=target_ae['enc_dim'], data_gen=stack_tsgen_t)
i_optimizer_lstmrnn = bayesian_opt(
    init_config=i_config, nn_config=i_nn, type_nn='rnn', type_rnn=1,
    parameters=lstm_rnn_params, enc_dim=target_ae['enc_dim'], data_gen=stack_tsgen_t)
i_optimizer_grurnn = bayesian_opt(
    init_config=i_config, nn_config=i_nn, type_nn='rnn', type_rnn=2,
    parameters=gru_rnn_params, enc_dim=target_ae['enc_dim'], data_gen=stack_tsgen_t)

s_rnn_opt = i_optimizer_srnn.optimizer(n_init_explore_point=50, n_bayesian_iterations=50, log_json=True)
tf.keras.backend.clear_session()

lstm_rnn_opt = i_optimizer_lstmrnn.optimizer(n_init_explore_point=50, n_bayesian_iterations=50, log_json=True)
tf.keras.backend.clear_session()

gru_rnn_opt = i_optimizer_grurnn.optimizer(n_init_explore_point=50, n_bayesian_iterations=50, log_json=True)
tf.keras.backend.clear_session()

# %% fitting of the test products dataset on optimized auto-encoder

# %% construction of stacked dataset with test products

# %% predict with optimized recurrent NN of stacked dataset with test products
