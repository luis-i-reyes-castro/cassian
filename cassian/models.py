"""
@author: Luis I. Reyes-Castro

COPYRIGHT

All contributions by Luis I. Reyes-Castro:
Copyright (c) 2017, Luis Ignacio Reyes Castro.
All rights reserved.
"""

import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils.vis_utils import plot_model

from .data_management import Dataset
from .batching import BatchSpecifications, BatchSample
from .core import NonlinearPID
from .convenience import exists_file, ensure_directory
from .convenience import serialize, de_serialize
from .convenience import get_timestamp, move_date
from .convenience import save_df_to_excel

# =====================================================================================
class CassianModel :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    OUTPUT_DIR   = 'trained-models/'
    OUTPUT_FILE  = 'store-[STORE-ID]_model_[TIME]_R-[REG]_L-[LR].pkl'
    WEIGHTS_FILE = 'store-[STORE-ID]_weights_[TIME]_R-[REG]_L-[LR].h5'
    RESULTS_DIR  = 'results/'
    RESULTS_FILE = 'store-[STORE-ID]_results.pkl'
    SUMMARY_FILE = 'store-[STORE-ID]_summary.xlsx'
    TB_LOG_DIR   = 'tensorboard-logs/' + OUTPUT_FILE

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, dataset_filename, batch_size,
                        timesteps = 90,
                        dense_layer_sizes = [ 256, 128 ],
                        NLPID_layer_sizes = [ 256, 256 ],
                        regularization = 1E-4,
                        learning_rate = 1E-3 ) :

        print( 'Current task: Loading Dataset instance' )
        if not exists_file( dataset_filename) :
            raise ValueError( 'Did not find file:', str(dataset_filename))

        self.dataset_filename = dataset_filename
        self.dataset = Dataset.load( self.dataset_filename)

        print( 'Current task: Building CassianModel instance' )
        self.store_id  = self.dataset.store_id
        self.timestamp = get_timestamp()

        self.batch_size        = batch_size
        self.timesteps         = timesteps
        self.dense_layer_sizes = dense_layer_sizes
        self.NLPID_layer_sizes = NLPID_layer_sizes
        self.regularization    = regularization
        # self.dropout_u         = None
        # self.dropout_x         = None
        self.learning_rate     = learning_rate

        self.regularize_hard = lambda : regularizers.l1( regularization )
        self.regularize_soft = lambda : regularizers.l1( 0.001 * regularization )

        self.loss_functions     = {}
        self.validation_metrics = {}

        # -----------------------------------------------------------------------------
        # Builds the U-input and X-input layers

        U_vecs_shape = ( batch_size, self.dataset.vec_dim)
        U_vecs       = Input( batch_shape = U_vecs_shape, name = 'U_vectors')

        X_vecs_shape = ( batch_size, None, self.dataset.ts_dim)
        X_vecs       = Input( batch_shape = X_vecs_shape, name = 'X_t_vectors')

        # Builds a stack of dense layers

        highest_U_vecs = U_vecs

        for ( i, layer_size) in enumerate( self.dense_layer_sizes) :
            layer_name = 'Feedforward-' + str(i+1)
            layer = Dense( name = layer_name,
                           units = layer_size,
                           activation = 'softsign',
                           kernel_regularizer = self.regularize_hard(),
                           bias_regularizer = self.regularize_soft() )
            highest_U_vecs = layer( highest_U_vecs )
            # highest_U_vecs = Dropout( name = 'Dropout-FF-' + str(i+1),
            #                           rate = self.dropout_u )( highest_U_vecs )

        # Builds a stack of NonlinearPID layers

        highest_X_vecs = X_vecs

        for ( i, layer_size) in enumerate( self.NLPID_layer_sizes) :
            layer_name = 'NonlinearPID-'+ str(i+1)
            layer = NonlinearPID( name = layer_name,
                                  units = layer_size,
                                  return_sequences = True,
                                  mat_R_z_regularizer = self.regularize_hard(),
                                  vec_b_z_regularizer = self.regularize_soft(),
                                  mat_R_0_regularizer = self.regularize_hard(),
                                  vec_b_0_regularizer = self.regularize_soft(),
                                  mat_R_b_regularizer = self.regularize_hard(),
                                  mat_W_p_regularizer = self.regularize_hard(),
                                  mat_W_i_regularizer = self.regularize_hard(),
                                  mat_W_d_regularizer = self.regularize_hard() )
            highest_X_vecs = layer( [ highest_U_vecs, highest_X_vecs] )

        # Builds the list of outputs.
        # The first output is the mean of a Poisson random variable
        # (a strictly positive real number) and is trained with Poisson loss.
        # The second output is binary and is trained with binary cross-entropy.
        # The last five outputs are each probability vectors (i.e. softmax)
        # trained with sparse categorical cross-entropy

        outputs_list = []

        dense_layer_names = [ 'Out-' + str(i+1) for i in range(7) ]
        layer_names       = [ 'Sold', 'Is_On_Sale',
                              'Replenished', 'Returned', 'Trashed',
                              'Found', 'Missing' ]
        layer_dims        = [ 1, 1,
                              self.dataset.z_replenished_dim,
                              self.dataset.z_returned_dim,
                              self.dataset.z_trashed_dim,
                              self.dataset.z_found_dim,
                              self.dataset.z_missing_dim ]
        layer_activations = [ K.exp, K.sigmoid ] + 5 * [ 'softmax' ]

        for ( dense_layer_name, layer_name, layer_dim, layer_activation) in \
            zip( dense_layer_names, layer_names, layer_dims, layer_activations) :

            dense_layer = Dense( name = dense_layer_name,
                                 input_dim = self.NLPID_layer_sizes[-1],
                                 units = layer_dim,
                                 activation = layer_activation,
                                 kernel_regularizer = self.regularize_hard(),
                                 bias_regularizer = self.regularize_hard() )

            output_tensor = TimeDistributed( name = layer_name,
                                             layer = dense_layer )( highest_X_vecs )

            outputs_list.append( output_tensor)

        # -----------------------------------------------------------------------------
        self.model = Model( inputs = [ U_vecs, X_vecs], outputs = outputs_list)
        self.compile_model()

        # -----------------------------------------------------------------------------
        ensure_directory( self.OUTPUT_DIR)

        self.output_file = self.OUTPUT_DIR + self.OUTPUT_FILE
        self.output_file = self.output_file.replace( '[STORE-ID]',
                                                     str(self.store_id))
        self.output_file = self.output_file.replace( '[TIME]',
                                                     self.timestamp)
        self.output_file = self.output_file.replace( '[REG]',
                                                     str(self.regularization))
        self.output_file = self.output_file.replace( '[LR]',
                                                     str(self.learning_rate))

        self.weights_file = self.OUTPUT_DIR + self.WEIGHTS_FILE
        self.weights_file = self.weights_file.replace( '[STORE-ID]',
                                                       str(self.store_id))
        self.weights_file = self.weights_file.replace( '[TIME]',
                                                       self.timestamp)
        self.weights_file = self.weights_file.replace( '[REG]',
                                                       str(self.regularization))
        self.weights_file = self.weights_file.replace( '[LR]',
                                                       str(self.learning_rate))

        self.tb_log_dir = self.TB_LOG_DIR.replace( '[STORE-ID]',
                                                   str(self.store_id))
        self.tb_log_dir = self.tb_log_dir.replace( '[TIME]',
                                                   self.timestamp)
        self.tb_log_dir = self.tb_log_dir.replace( '[REG]',
                                                   str(self.regularization))
        self.tb_log_dir = self.tb_log_dir.replace( '[LR]',
                                                   str(self.learning_rate))

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compile_model( self) :

        def root_mean_squared_error( y_true, y_pred) :
            return K.sqrt( K.mean( K.square(y_pred - y_true), axis=-1) )

        self.loss_functions                = {}
        self.loss_functions['Sold']        = 'poisson'
        self.loss_functions['Is_On_Sale']  = 'binary_crossentropy'
        self.loss_functions['Replenished'] = 'sparse_categorical_crossentropy'
        self.loss_functions['Returned']    = 'sparse_categorical_crossentropy'
        self.loss_functions['Trashed']     = 'sparse_categorical_crossentropy'
        self.loss_functions['Found']       = 'sparse_categorical_crossentropy'
        self.loss_functions['Missing']     = 'sparse_categorical_crossentropy'

        self.validation_metrics                = {}
        self.validation_metrics['Sold']        = root_mean_squared_error
        self.validation_metrics['Is_On_Sale']  = 'accuracy'
        self.validation_metrics['Replenished'] = 'categorical_accuracy'
        self.validation_metrics['Returned']    = 'categorical_accuracy'
        self.validation_metrics['Trashed']     = 'categorical_accuracy'
        self.validation_metrics['Found']       = 'categorical_accuracy'
        self.validation_metrics['Missing']     = 'categorical_accuracy'

        self.optimizer = optimizers.Adam( lr = self.learning_rate,
                                          decay = 1E-4,
                                          beta_1 = 0.9,
                                          beta_2 = 0.998 )

#        self.optimizer = optimizers.SGD( lr = self.learning_rate,
#                                         momentum = 0.9,
#                                         nesterov = True )

        self.model.compile( loss = self.loss_functions,
                            metrics = self.validation_metrics,
                            optimizer = self.optimizer )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def save_model( self) :

        output_dict = {}
        output_dict['dataset_filename']  = self.dataset_filename
        output_dict['batch_size']        = self.batch_size
        output_dict['timesteps']         = self.timesteps
        output_dict['dense_layer_sizes'] = self.dense_layer_sizes
        output_dict['NLPID_layer_sizes'] = self.NLPID_layer_sizes
        output_dict['weights_filename']  = self.weights_file

        print( 'Saving CassianModel instance to file:', self.output_file)
        serialize( output_dict, self.output_file)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def save_weights( self) :

        print( 'Saving CassianModel weights to file:', self.weights_file)
        self.model.save_weights( self.weights_file)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def load( cassian_model_file) :

        print( 'Loading CassianModel instance from file:', cassian_model_file)
        if not exists_file( cassian_model_file) :
            raise ValueError( 'Did not find file:', str(cassian_model_file))

        input_dict = de_serialize( cassian_model_file)

        cassian = CassianModel( input_dict['dataset_filename'],
                                input_dict['batch_size'],
                                input_dict['timesteps'],
                                input_dict['dense_layer_sizes'],
                                input_dict['NLPID_layer_sizes'] )

        cassian.model.load_weights( input_dict['weights_filename'] )

        cassian.compile_model()

        return cassian

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def plot_model( self, filename = 'Model-Architecture.jpg') :

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def train_on_dataset( self, epochs = 1, patience = 100, workers = 4) :

        print( 'Current task: Training for ' + str(epochs) + ' epochs' )

        def batch_generator( dataset) :

            batch_specs            = BatchSpecifications()
            batch_specs.batch_size = self.batch_size
            batch_specs.timesteps  = self.timesteps
            batch_specs.vec_dim    = dataset.vec_dim
            batch_specs.ts_dim     = dataset.ts_dim

            batch_specs.ts_replenished_dim = dataset.z_replenished_dim
            batch_specs.ts_returned_dim    = dataset.z_returned_dim
            batch_specs.ts_trashed_dim     = dataset.z_trashed_dim
            batch_specs.ts_found_dim       = dataset.z_found_dim
            batch_specs.ts_missing_dim     = dataset.z_missing_dim

            batch_sample = BatchSample( batch_specs)

            while True :

                batch_skus = np.random.choice( a = dataset.list_of_skus,
                                               p = dataset.list_of_sku_probs,
                                               size = self.batch_size )

                random_seeds = np.random.rand( self.batch_size)

                for ( sample_i, sku) in enumerate( batch_skus ) :

                    dataset(sku).get_sample( timesteps = self.timesteps,
                                             seed = random_seeds[sample_i],
                                             batch_sample = batch_sample,
                                             sample_index = sample_i)

                yield [ batch_sample.inputs, batch_sample.targets ]

            return

        pieces_per_epoch = 1
        epochs           = pieces_per_epoch * epochs
        patience         = pieces_per_epoch * patience

        ( dataset_train, dataset_valid) = self.dataset.split(0.68)

        steps_per_epoch_train = dataset_train.num_timesteps \
                              // ( self.batch_size * self.timesteps * pieces_per_epoch )
        steps_per_epoch_valid = dataset_valid.num_timesteps \
                              // ( self.batch_size * self.timesteps * pieces_per_epoch )

        ensure_directory(self.tb_log_dir)

        callbacks = [ ModelCheckpoint( self.weights_file,
                                       monitor = 'Sold_root_mean_squared_error',
                                       mode = 'min',
                                       save_weights_only = True,
                                       save_best_only = True),
                      EarlyStopping( monitor = 'Sold_root_mean_squared_error',
                                     patience = patience, mode = 'min'),
                      TensorBoard( log_dir = self.tb_log_dir, write_graph = False) ]

        self.save_model()

        self.model.fit_generator( generator = batch_generator( dataset_train),
                                  validation_data = batch_generator( dataset_valid),
                                  epochs = epochs,
                                  steps_per_epoch = steps_per_epoch_train,
                                  validation_steps = steps_per_epoch_valid,
                                  workers = workers,
                                  callbacks = callbacks,
                                  pickle_safe = True )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_predictions( self) :

        print( 'Current task: Evaluating predictions' )

        num_batches = self.dataset.num_skus // self.batch_size
        if self.dataset.num_skus % self.batch_size > 0 :
            num_batches += 1

        X_vec = np.zeros( ( self.batch_size, self.dataset.vec_dim) )
        X_ts  = np.zeros( ( self.batch_size, self.timesteps, self.dataset.ts_dim) )

        inputs  = [ [ X_vec.copy(), X_ts.copy() ] for _ in range(num_batches) ]
        outputs = [ None for _ in range(num_batches) ]

        curr_batch   = 0
        curr_sample  = 0
        sku_at_location = np.zeros( ( num_batches, self.batch_size), dtype = int)

        predictions  = { sku : None for sku in self.dataset() }
        summary      = self.dataset.info_description.copy()

        for sku in self.dataset() :

            sku_at_location[ curr_batch, curr_sample] = sku

            ( x_vec, x_ts) = \
            self.dataset(sku).get_most_recent_inputs( self.timesteps)

            inputs[curr_batch][0][ curr_sample, :] = x_vec
            inputs[curr_batch][1][ curr_sample, :, :] = x_ts

            curr_sample = ( curr_sample + 1 ) % self.batch_size
            curr_batch += 1 if curr_sample == 0 else 0

        for i in range(num_batches) :

            print( 'Evaluating batch', str(i+1), 'of', str(num_batches))
            outputs[i] = self.model.predict( inputs[i], batch_size = self.batch_size)

            for j in range(self.batch_size) :

                sku = sku_at_location[i,j]
                if not sku :
                    break
                else :
                    print( 'Collecting predictions for SKU:', str(sku))

                df_index = self.dataset(sku).timeseries.index[ -self.timesteps: ]
                df_cols  = [ 'Initial_Stock', 'Sold', 'Expected_Sales']

                predictions[sku] = pd.DataFrame( data = np.NAN,
                                                 index = df_index, columns = df_cols)

                predictions[sku]['Initial_Stock'] = \
                self.dataset(sku).timeseries.iloc[ -self.timesteps: ]['STOCK_INITIAL']

                predictions[sku]['Sold'] = \
                self.dataset(sku).timeseries.iloc[ -self.timesteps: ]['SOLD']

                predicted_sold = outputs[i][0][ j, :, 0]

                second_day = predictions[sku].index[1]
                last_day   = predictions[sku].index[-1]

                predictions[sku].loc[ second_day : last_day, 'Expected_Sales'] = \
                predicted_sold[:-1]

                summary.loc[ sku, 'Initial_Stock'] = \
                self.dataset(sku).timeseries.loc[ last_day, 'STOCK_FINAL']

                last_day = move_date( date = last_day, delta_days = +1)

                predictions[sku].loc[ last_day, 'Expected_Sales'] = \
                predicted_sold[-1]

                summary.loc[ sku, 'Expected_Sales'] = predicted_sold[-1]

        summary['Days_of_Stock'] = \
        summary['Initial_Stock'] / summary['Expected_Sales']
        summary.sort_values( by = 'Expected_Sales', ascending = False, inplace = True)

        results_dict                = {}
        results_dict['predictions'] = predictions
        results_dict['summary']     = summary

        ensure_directory( self.RESULTS_DIR)

        results_file = self.RESULTS_DIR \
                     + self.RESULTS_FILE.replace( '[STORE-ID]', str(self.store_id))

        print( 'Saving results to file:', results_file)
        serialize( results_dict, results_file)

        summary_file = self.RESULTS_DIR \
                     + self.SUMMARY_FILE.replace( '[STORE-ID]', str(self.store_id))

        print( 'Saving summary to file:', summary_file)
        save_df_to_excel( summary, summary_file)

        return predictions, summary
