"""
@author: Luis I. Reyes Castro
"""

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.recurrent import SimpleRNN
from keras.models import Model
from keras.utils.vis_utils import plot_model
from .layers import VectorDependentGatedRNN

# =====================================================================================
class CassianModel :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, dataset, vector_embedding_dim = 16,
                                 layer_sizes = [ 256, 256, 256] ) :

        def exponential( tensor) :
            return K.exp(tensor)

        def zero_one_softsign( tensor) :
            return 0.5 + 0.5 * K.softsign( 2.0 * tensor )

        self.dataset          = dataset
        self.specs            = dataset.batch_specs

        if self.specs is None :
            raise ValueError( 'Dataset sampler method has not been setup' )

        self.embed_dim        = vector_embedding_dim
        self.layer_sizes      = layer_sizes
        self.learnable_layers = []
        self.outputs_list     = []
        self.loss_functions   = {}

        # -----------------------------------------------------------------------------
        # Builds the input layers and the dimensionality reduction layer

        self.X_vecs_shape = ( self.specs.batch_size, self.specs.vec_dim)
        self.X_ts_shape   = ( self.specs.batch_size, None, self.specs.ts_dim)
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
        layer_activations = [ exponential, zero_one_softsign ]
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
        layer_dims  = [ self.specs.ts_replenished_dim,
                        self.specs.ts_returned_dim,
                        self.specs.ts_trashed_dim,
                        self.specs.ts_found_dim,
                        self.specs.ts_missing_dim ]
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
    def plot_model( self, filename = 'Cassian.png') :

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)
