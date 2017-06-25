"""
@author: Luis I. Reyes-Castro

COPYRIGHT

All contributions by Luis I. Reyes-Castro:
Copyright (c) 2017, Luis Ignacio Reyes Castro.
All rights reserved.
"""

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.recurrent import SimpleRNN
from keras.models import Model
from keras.utils.vis_utils import plot_model

from .data_management import Dataset, BatchSpecifications, BatchSample
from .layers import VectorDependentGatedRNN
from .convenience import move_date

# =====================================================================================
class CassianModel :

    dataset = Dataset()

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, dataset, batch_size = 32, timesteps = 90,
                        vector_embedding_dim = 16,
                        layer_sizes = [ 128, 128, 128] ) :

        self.dataset    = dataset
        self.batch_size = batch_size
        self.timesteps  = timesteps

        self.embed_dim        = vector_embedding_dim
        self.layer_sizes      = layer_sizes
        self.learnable_layers = []
        self.outputs_list     = []
        self.loss_functions   = {}
        self.steps_per_epoch  = dataset.num_timesteps % ( batch_size * timesteps )

        # -----------------------------------------------------------------------------
        # Builds the input layers and the dimensionality reduction layer

        self.X_vecs_shape = ( batch_size, self.dataset.vec_dim)
        self.X_ts_shape   = ( batch_size, None, self.dataset.ts_dim)

        self.X_vecs       = Input( batch_shape = self.X_vecs_shape,
                                   name = 'Product_Vectors')
        self.X_ts         = Input( batch_shape = self.X_ts_shape,
                                   name = 'Product_TS')

        layer_name = 'Dim_Reduction'
        self.embedding = Dense( units = self.embed_dim,
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

        layer_activations = [ K.softplus, zero_one_softsign ]
        layer_losses = [ 'poisson', 'binary_crossentropy' ]

        for ( layer_name, layer_activation, layer_loss) in \
            zip( layer_names, layer_activations, layer_losses) :

            layer_object = SimpleRNN( name = layer_name,
                                      units = 1,
                                      activation = layer_activation,
                                      return_sequences = True)
            output_tensor = layer_object( self.layer_outputs[-1] )

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

            layer_object = SimpleRNN( name = layer_name,
                                      units = layer_dim,
                                      activation = 'softmax',
                                      return_sequences = True)
            output_tensor = layer_object( self.layer_outputs[-1] )

            self.learnable_layers.append( layer_name)
            self.outputs_list.append( output_tensor)
            self.loss_functions[layer_name] = layer_losses

        self.model = Model( inputs = [ self.X_vecs, self.X_ts],
                            outputs = self.outputs_list )

        self.model.compile( optimizer = 'adam', loss = self.loss_functions)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def as_dictionary( self) :

        return self.__dict__

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def copy_weights( self, original_cassian) :

        for layer in self.learnable_layers :

            original_weights = original_cassian.model.get_layer(layer).get_weights()
            self.model.get_layer(layer).set_weights( original_weights )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def plot_model( self, filename = 'Cassian.png') :

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def train_on_dataset( self, epochs = 1, workers = 4) :

        self.model.fit_generator( generator = CassianBatchGenerator(self),
                                  steps_per_epoch = self.steps_per_epoch,
                                  epochs = epochs,
                                  workers = workers, pickle_safe = True)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_predictions( self) :

        num_batches = self.dataset.num_skus // self.batch_size
        if self.dataset.num_skus % self.batch_size > 0 :
            num_batches += 1

        X_vec = np.zeros( ( self.batch_size, self.dataset.vec_dim) )
        X_ts  = np.zeros( ( self.batch_size, self.timesteps, self.dataset.ts_dim) )

        inputs  = [ [ X_vec.copy(), X_ts.copy() ] for _ in range(num_batches) ]
        outputs = [ None for _ in range(num_batches) ]
#        summary = self.dataset.info_description.copy()

        curr_batch   = 0
        curr_sample  = 0
        sku_location = {}
        pred_sold    = {}

        for sku in self.dataset() :

            ( x_vec, x_ts) = \
            self.dataset(sku).get_most_recent_inputs( self.timesteps)

            inputs[curr_batch][0][ curr_sample, :] = x_vec
            inputs[curr_batch][1][ curr_sample, :, :] = x_ts

            sku_location[sku] = ( curr_batch, curr_sample)

            curr_sample = ( curr_sample + 1 ) % self.batch_size
            curr_batch += 1 if curr_sample == 0 else 0

            pred_sold[sku] = self.dataset(sku).timeseries[ ['SOLD'] ]
            pred_sold[sku] = pred_sold[sku].iloc[ -self.timesteps : ]

        for i in range(num_batches) :
            outputs[i] = self.model.predict( inputs[i], batch_size = self.batch_size)

        for sku in self.dataset() :

            batch  = sku_location[sku][0]
            sample = sku_location[sku][1]

            predicted_ts = outputs[batch][0][ sample, :, 0]

            start = pred_sold[sku].index[1]
            end   = pred_sold[sku].index[-1]
            pred_sold[sku].loc[ start : end, 'PRED'] = predicted_ts[:-1]

            end = move_date( date = end, delta_days = +1)
            pred_sold[sku].loc[ end, 'PRED'] = predicted_ts[-1]

        return pred_sold

# =====================================================================================
def CassianBatchGenerator( cassian) :

    dataset = cassian.dataset

    batch_specs            = BatchSpecifications()
    batch_specs.batch_size = cassian.batch_size
    batch_specs.timesteps  = cassian.timesteps
    batch_specs.vec_dim    = dataset.vec_dim
    batch_specs.ts_dim     = dataset.ts_dim

    batch_specs.ts_replenished_dim = dataset.ts_replenished_dim
    batch_specs.ts_returned_dim    = dataset.ts_returned_dim
    batch_specs.ts_trashed_dim     = dataset.ts_trashed_dim
    batch_specs.ts_found_dim       = dataset.ts_found_dim
    batch_specs.ts_missing_dim     = dataset.ts_missing_dim

    batch_sample = BatchSample( batch_specs)

    while True :

            batch_skus = np.random.choice( a = dataset.list_of_skus,
                                           p = dataset.list_of_sku_probs,
                                           size = cassian.batch_size )

            for sku in batch_skus :
                ( inputs, targets) = dataset(sku).get_sample( cassian.timesteps)
                batch_sample.include_sample( inputs, targets)

            yield [ batch_sample.inputs, batch_sample.targets ]

    return
