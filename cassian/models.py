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
from keras.layers import Input, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils.vis_utils import plot_model

from .data_management import Dataset
from .batching import BatchSpecifications, BatchSample
from .core import HybridUnit
from .convenience import exists_file, ensure_directory
from .convenience import serialize, de_serialize
from .convenience import get_timestamp, move_date
from .convenience import save_df_to_excel

# =====================================================================================
class CassianModel :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    OUTPUT_DIR   = 'trained-models/'
    OUTPUT_FILE  = 'store-[STORE-ID]_model_[TIMESTAMP].pkl'
    WEIGHTS_FILE = 'store-[STORE-ID]_weights_[TIMESTAMP].h5'
    RESULTS_DIR  = 'results/'
    RESULTS_FILE = 'store-[STORE-ID]_results.pkl'
    SUMMARY_FILE = 'store-[STORE-ID]_summary.xlsx'
    TB_LOG_DIR   = 'tensorboard-logs/store-[STORE-ID]/[TIMESTAMP]'

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, dataset_filename, batch_size, timesteps,
                        dense_layer_sizes = [ 512 ],
                        SGHU_layer_sizes = [ 512 ] ) :

        print( 'Current task: Loading Dataset instance' )
        if not exists_file( dataset_filename) :
            raise ValueError( 'Did not find file:', str(dataset_filename))

        self.dataset_filename = dataset_filename
        self.dataset = Dataset.load( self.dataset_filename)

        print( 'Current task: Building CassianModel instance' )
        self.store_id  = self.dataset.store_id
        self.timestamp = get_timestamp()

        self.batch_size      = batch_size
        self.timesteps       = timesteps
        self.steps_per_epoch = self.dataset.num_timesteps // ( batch_size * timesteps )

        self.dense_layer_sizes = dense_layer_sizes
        self.SGHU_layer_sizes  = SGHU_layer_sizes

        self.outputs_list   = []
        self.loss_functions = {}

        # -----------------------------------------------------------------------------
        # Builds the input layers and the dimensionality reduction layer

        self.X_vecs_shape = ( batch_size, self.dataset.vec_dim)
        self.X_ts_shape   = ( batch_size, None, self.dataset.ts_dim)

        X_vecs = Input( batch_shape = self.X_vecs_shape, name = 'Product_Vectors')
        X_ts   = Input( batch_shape = self.X_ts_shape, name = 'Product_TS')

        # -----------------------------------------------------------------------------
        # Builds a stack of several layers of dimensionality reduction

        last_output_vector = X_vecs

        for ( i, layer_size) in enumerate( self.dense_layer_sizes) :
            layer_name = 'Feedforward-' + str(i+1)
            layer = Dense( name = layer_name, units = layer_size,
                           activation = 'softsign')
            last_output_vector = layer( last_output_vector )
            # shape = ( batch_size, None, dim_reduction_layer_dim)

        # -----------------------------------------------------------------------------
        # Builds a stack of several layers of Vector-dependent gated RNNs

        last_output_ts = X_ts

        for ( i, layer_size) in enumerate( self.SGHU_layer_sizes) :
            layer_name = 'SingleGateHybridUnit-'+ str(i+1)
            layer = HybridUnit( name = layer_name, units = layer_size,
                                return_sequences = True)
            last_output_ts = layer( [ last_output_vector, last_output_ts] )
            # shape = ( batch_size, None, layer_dim)

        # -----------------------------------------------------------------------------
        # Builds the first two output timeseries: sold and is-on-sale
        # The first output is the mean of a Poisson random variable
        # (a strictly positive real number) and is trained with Poisson loss
        # The second output is binary and is trained with binary cross-entropy

        dense_layer_names = [ 'Out-1', 'Out-2']
        layer_names       = [ 'Sold', 'Is_On_Sale' ]

        def zero_one_softsign( tensor) :
            return 0.5 + 0.5 * K.softsign( 4.0 * tensor )

        layer_activations = [ K.exp, zero_one_softsign ]
        layer_losses      = [ 'poisson', 'binary_crossentropy' ]

        for ( dense_layer_name, layer_name, layer_activation, layer_loss) in \
            zip( dense_layer_names, layer_names, layer_activations, layer_losses) :

            dense_layer = Dense( name = dense_layer_name,
                                 input_dim = self.SGHU_layer_sizes[-1],
                                 units = 1,
                                 activation = layer_activation)

            output_tensor = \
            TimeDistributed( name = layer_name, layer = dense_layer)( last_output_ts )

            self.outputs_list.append( output_tensor)
            self.loss_functions[layer_name] = layer_loss

        # -----------------------------------------------------------------------------
        # Builds the last five output timeseries, each of which is a
        # probability softmax (i.e. a softmax) trained with
        # sparse categorical cross-entropy

        dense_layer_names = [ 'Out-' + str(i+3) for i in range(5)  ]
        layer_names = [ 'Replenished', 'Returned', 'Trashed', 'Found', 'Missing' ]
        layer_dims  = [ self.dataset.z_replenished_dim,
                        self.dataset.z_returned_dim,
                        self.dataset.z_trashed_dim,
                        self.dataset.z_found_dim,
                        self.dataset.z_missing_dim ]
        layer_losses = 'sparse_categorical_crossentropy'

        for ( dense_layer_name, layer_name, layer_dim) in \
            zip( dense_layer_names, layer_names, layer_dims) :

            dense_layer = Dense( name = dense_layer_name,
                                 input_dim = self.SGHU_layer_sizes[-1],
                                 units = layer_dim,
                                 activation = 'softmax')

            output_tensor = \
            TimeDistributed( name = layer_name, layer = dense_layer)( last_output_ts )

            self.outputs_list.append( output_tensor)
            self.loss_functions[layer_name] = layer_losses

        # -----------------------------------------------------------------------------
        self.validation_metrics                = {}
        self.validation_metrics['Sold']        = 'mae'
        self.validation_metrics['Is_On_Sale']  = 'accuracy'
        self.validation_metrics['Replenished'] = 'categorical_accuracy'
        self.validation_metrics['Returned']    = 'categorical_accuracy'
        self.validation_metrics['Trashed']     = 'categorical_accuracy'
        self.validation_metrics['Found']       = 'categorical_accuracy'
        self.validation_metrics['Missing']     = 'categorical_accuracy'

        # -----------------------------------------------------------------------------
        self.model = Model( inputs = [ X_vecs, X_ts],
                            outputs = self.outputs_list )
        self.compile_model()

        # -----------------------------------------------------------------------------
        ensure_directory( self.OUTPUT_DIR)

        self.output_file = self.OUTPUT_DIR + self.OUTPUT_FILE
        self.output_file = self.output_file.replace( '[STORE-ID]', str(self.store_id))
        self.output_file = self.output_file.replace( '[TIMESTAMP]', self.timestamp)

        self.weights_file = self.OUTPUT_DIR + self.WEIGHTS_FILE
        self.weights_file = self.weights_file.replace( '[STORE-ID]', str(self.store_id))
        self.weights_file = self.weights_file.replace( '[TIMESTAMP]', self.timestamp)

        self.tb_log_dir = self.TB_LOG_DIR.replace( '[STORE-ID]', str(self.store_id))
        self.tb_log_dir = self.tb_log_dir.replace( '[TIMESTAMP]', self.timestamp)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compile_model( self) :

        self.optimizer = optimizers.SGD( lr = 0.001,
                                         momentum = 0.9,
                                         nesterov = True )

#        self.optimizer = optimizers.Adamax( lr = 0.001,
#                                            beta_1 = 0.9,
#                                            beta_2 = 0.99,
#                                            epsilon = 1E-6 )

        self.model.compile( optimizer = self.optimizer,
                            loss = self.loss_functions,
                            metrics = self.validation_metrics )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def save_model( self) :

        output_dict = {}
        output_dict['dataset_filename']  = self.dataset_filename
        output_dict['batch_size']        = self.batch_size
        output_dict['timesteps']         = self.timesteps
        output_dict['dense_layer_sizes'] = self.dense_layer_sizes
        output_dict['SGHU_layer_sizes']  = self.SGHU_layer_sizes
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
                                input_dict['SGHU_layer_sizes'] )

        cassian.model.load_weights( input_dict['weights_filename'] )

        cassian.compile_model()

        return cassian

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def plot_model( self, filename = 'Model-Architecture.jpg') :

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def train_on_dataset( self, epochs = 1, patience = 10, workers = 4) :

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

        ( dataset_train, dataset_valid) = self.dataset.split(0.75)

        pieces_per_epoch       = 10
        actual_patience        = patience * pieces_per_epoch
        actual_epochs          = epochs * pieces_per_epoch
        actual_steps_per_epoch = self.steps_per_epoch // pieces_per_epoch

        ensure_directory(self.tb_log_dir)

        callbacks = [ ModelCheckpoint( self.weights_file,
                                       monitor = 'val_Sold_mean_absolute_error',
                                       mode = 'min',
                                       save_weights_only = True,
                                       save_best_only = True),
                      EarlyStopping( monitor = 'Sold_mean_absolute_error',
                                     patience = actual_patience, mode = 'min'),
                      TensorBoard( log_dir = self.tb_log_dir, write_graph = False) ]

        self.save_model()

        self.model.fit_generator( generator = batch_generator( dataset_train),
                                  validation_data = batch_generator( dataset_valid),
                                  epochs = actual_epochs,
                                  steps_per_epoch = actual_steps_per_epoch,
                                  validation_steps = actual_steps_per_epoch,
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
        summary.sort_values( by = 'Days_of_Stock', ascending = False, inplace = True)

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
