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
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from .data_management import Dataset, BatchSpecifications, BatchSample
from .layers import VectorDependentGatedRNN
from .convenience import exists_file, ensure_directory
from .convenience import serialize, de_serialize
from .convenience import move_date
from .convenience import save_df_to_excel

# =====================================================================================
class CassianModel :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    OUTPUT_DIR   = '/home/luis/cassian/trained-models/'
    OUTPUT_FILE  = 'store-[STORE-ID]_model.pkl'
    WEIGHTS_FILE = 'store-[STORE-ID]_weights.h5'
    RESULTS_DIR  = '/home/luis/cassian/results/'
    RESULTS_FILE = 'store-[STORE-ID]_results.pkl'
    SUMMARY_FILE = 'store-[STORE-ID]_summary.xlsx'

    dataset_filename = None
    dataset          = Dataset()

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, dataset_filename, batch_size = 32, timesteps = 90,
                        vector_embedding_dim = 64,
                        layer_sizes = [ 256, 256, 256] ) :

        print( 'Current task: Loading Dataset instance' )
        if not exists_file( dataset_filename) :
            raise ValueError( 'Did not find file:', str(dataset_filename))

        self.dataset_filename = dataset_filename
        self.dataset = Dataset.load( self.dataset_filename)

        print( 'Current task: Building CassianModel instance' )
        self.store_id        = self.dataset.store_id
        self.batch_size      = batch_size
        self.timesteps       = timesteps
        self.steps_per_epoch = self.dataset.num_timesteps // ( batch_size * timesteps )

        self.vector_embedding_dim = vector_embedding_dim
        self.layer_sizes          = layer_sizes
        self.learnable_layers     = []
        self.outputs_list         = []
        self.loss_functions       = {}

        # -----------------------------------------------------------------------------
        # Builds the input layers and the dimensionality reduction layer

        self.X_vecs_shape = ( batch_size, self.dataset.vec_dim)
        self.X_ts_shape   = ( batch_size, None, self.dataset.ts_dim)

        self.X_vecs       = Input( batch_shape = self.X_vecs_shape,
                                   name = 'Product_Vectors')
        self.X_ts         = Input( batch_shape = self.X_ts_shape,
                                   name = 'Product_TS')

        layer_name = 'Dim_Reduction'
        self.embedding = Dense( units = self.vector_embedding_dim,
                                activation = 'softsign',
                                name = layer_name )( self.X_vecs)
        self.learnable_layers.append( layer_name )

        # -----------------------------------------------------------------------------
        # Builds a stack of several layers of Vector-dependent gated RNNs

        self.layer_outputs = [ self.X_ts ]

        for ( i, layer_size) in enumerate( self.layer_sizes) :

            layer_name = 'VDGRNN-'+ str(i+1)
            layer = VectorDependentGatedRNN( units = layer_size,
                                             learn_initial_state_bias = True,
                                             learn_initial_state_kernel = True,
                                             architecture = 'single-gate',
                                             name = layer_name)

            input_vectors = self.embedding
            # shape = ( batch_size, embedding_dim)
            input_timeseries = self.layer_outputs[-1]
            # shape = ( batch_size, None, timeseries_input_dim or prev_layer_dim)
            output_timeseries = layer( [ input_vectors, input_timeseries] )
            # shape = ( batch_size, None, layer_dim)

            self.learnable_layers.append( layer_name )
            self.layer_outputs.append( output_timeseries )

        self.layer_outputs = self.layer_outputs[1:]

        # -----------------------------------------------------------------------------
        # Builds the first two output timeseries: sold and is-on-sale
        # The first output is the mean of a Poisson random variable
        # (a strictly positive real number) and is trained with Poisson loss
        # The second output is binary and is trained with binary cross-entropy

        layer_names = [ 'Sold', 'Is_On_Sale' ]

        def zero_one_softsign( tensor) :
            return 0.5 + 0.5 * K.softsign( 2.0 * tensor )

        layer_activations = [ K.exp, zero_one_softsign ]
        layer_losses = [ 'poisson', 'binary_crossentropy' ]

        for ( layer_name, layer_activation, layer_loss) in \
            zip( layer_names, layer_activations, layer_losses) :

            dense_layer = Dense( input_dim = self.layer_sizes[-1],
                                 units = 1,
                                 activation = layer_activation)

            output_tensor = \
            TimeDistributed( dense_layer, name = layer_name)( self.layer_outputs[-1] )

            self.learnable_layers.append( layer_name)
            self.outputs_list.append( output_tensor)
            self.loss_functions[layer_name] = layer_loss

        # -----------------------------------------------------------------------------
        # Builds the last five output timeseries, each of which is a
        # probability softmax (i.e. a softmax) trained with
        # sparse categorical cross-entropy

        layer_names = [ 'Replenished', 'Returned', 'Trashed', 'Found', 'Missing' ]
        layer_dims  = [ self.dataset.ts_replenished_dim,
                        self.dataset.ts_returned_dim,
                        self.dataset.ts_trashed_dim,
                        self.dataset.ts_found_dim,
                        self.dataset.ts_missing_dim ]
        layer_losses = 'sparse_categorical_crossentropy'

        for ( layer_name, layer_dim) in zip( layer_names, layer_dims) :

            dense_layer = Dense( input_dim = self.layer_sizes[-1],
                                 units = layer_dim,
                                 activation = 'softmax')

            output_tensor = \
            TimeDistributed( dense_layer, name = layer_name)( self.layer_outputs[-1] )

            self.learnable_layers.append( layer_name)
            self.outputs_list.append( output_tensor)
            self.loss_functions[layer_name] = layer_losses

        # -----------------------------------------------------------------------------
        self.model = Model( inputs = [ self.X_vecs, self.X_ts],
                            outputs = self.outputs_list )

        self.model.compile( optimizer = 'adam', loss = self.loss_functions)

        # -----------------------------------------------------------------------------
        ensure_directory( self.OUTPUT_DIR)

        self.output_file  = self.OUTPUT_DIR \
                          + self.OUTPUT_FILE.replace( '[STORE-ID]', str(self.store_id))
        self.weights_file = self.OUTPUT_DIR \
                          + self.WEIGHTS_FILE.replace( '[STORE-ID]', str(self.store_id))

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def as_dictionary( self) :

        return self.__dict__

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def save( self) :

        output_dict = {}
        output_dict['dataset_filename']     = self.dataset_filename
        output_dict['batch_size']           = self.batch_size
        output_dict['timesteps']            = self.timesteps
        output_dict['vector_embedding_dim'] = self.vector_embedding_dim
        output_dict['layer_sizes']          = self.layer_sizes
        output_dict['weights_filename']     = self.weights_file

        print( 'Saving CassianModel instance to file:', self.output_file)
        serialize( output_dict, self.output_file)

#        print( 'Saving CassianModel weights to file:', self.weights_file)
#        self.model.save_weights( self.weights_file)

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
                                input_dict['vector_embedding_dim'],
                                input_dict['layer_sizes'] )

        cassian.model.load_weights( input_dict['weights_filename'] )
        cassian.model.compile( optimizer = 'adam', loss = cassian.loss_functions)

        return cassian

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def copy_weights( self, original_model) :

        for layer in self.learnable_layers :

            original_weights = original_model.get_layer(layer).get_weights()

            self.model.get_layer(layer).set_weights( original_weights )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def plot_model( self, filename = 'Model-Architecture.png') :

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def train_on_dataset( self, epochs = 1, workers = 4) :

        def batch_generator() :

            batch_specs            = BatchSpecifications()
            batch_specs.batch_size = self.batch_size
            batch_specs.timesteps  = self.timesteps
            batch_specs.vec_dim    = self.dataset.vec_dim
            batch_specs.ts_dim     = self.dataset.ts_dim

            batch_specs.ts_replenished_dim = self.dataset.ts_replenished_dim
            batch_specs.ts_returned_dim    = self.dataset.ts_returned_dim
            batch_specs.ts_trashed_dim     = self.dataset.ts_trashed_dim
            batch_specs.ts_found_dim       = self.dataset.ts_found_dim
            batch_specs.ts_missing_dim     = self.dataset.ts_missing_dim

            batch_sample = BatchSample( batch_specs)

            while True :

                batch_skus = np.random.choice( a = self.dataset.list_of_skus,
                                               p = self.dataset.list_of_sku_probs,
                                               size = self.batch_size )

                random_seeds = np.random.rand( self.batch_size)

                for ( i, sku) in enumerate(batch_skus) :

                    ( inputs, targets) = \
                    self.dataset(sku).get_sample( self.timesteps, random_seeds[i])

                    batch_sample.include_sample( inputs, targets)

                yield [ batch_sample.inputs, batch_sample.targets ]

            return

        print( 'Current task: Training for ' + str(epochs) + ' epochs' )

        callbacks = [ ModelCheckpoint( self.weights_file,
                                       monitor = 'loss', mode = 'min',
                                       save_weights_only = True,
                                       save_best_only = True) ]

        self.model.fit_generator( generator = batch_generator(),
                                  steps_per_epoch = self.steps_per_epoch,
                                  epochs = epochs,
                                  workers = workers,
                                  callbacks = callbacks,
                                  pickle_safe = True )
        self.save()

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
