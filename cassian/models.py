"""
@author: Luis I. Reyes Castro
"""

import copy as cp
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils.vis_utils import plot_model
from .layers import VectorDependentGatedRNN

# =====================================================================================
class CassianModel :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, batch_specs, vector_embedding_dim = 16,
                                     layer_sizes = [ 256, 256, 256] ) :

        self.specs = cp.deepcopy(batch_specs)
#        self.batch_size       = self.specs.batch_size
#        self.vec_dim          = self.specs.vec_dim
#        self.ts_dim           = self.specs.ts_dim
        self.embed_dim        = vector_embedding_dim
        self.layer_sizes      = layer_sizes
        self.learnable_layers = []
        self.outputs_list     = []

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

        exp_activation      = lambda tensor : K.exp(tensor)
        zero_one_activation = lambda tensor : 0.5 + 0.5 * K.softsign( 2.0 * tensor )

        layer_name = 'Output-1'
        layer = Dense( units = 1,
                       input_dim = self.layer_sizes[-1],
                       activation = exp_activation,
                       name = layer_name )
        TD_name = 'Sold_TS'
        Y_hat_sales = \
        TimeDistributed( layer, name = TD_name )( self.layer_outputs[-1] )

        self.learnable_layers.append( TD_name )
        self.outputs_list.append( Y_hat_sales )

        layer_name = 'Output-2'
        layer = Dense( units = 1,
                       input_dim = self.layer_sizes[-1],
                       activation = zero_one_activation,
                       name = layer_name )
        TD_name = 'Is_on_sale_TS'
        Y_hat_is_on_sale = \
        TimeDistributed( layer, name = TD_name )( self.layer_outputs[-1] )

        self.learnable_layers.append( TD_name )
        self.outputs_list.append( Y_hat_is_on_sale )

        cat_out_index = tuple( range(5) )
        cat_out_names = [ 'Replenished', 'Returned', 'Trashed',
                          'Found', 'Missing' ]
        cat_out_dims  = [ self.specs.ts_replenished_dim,
                          self.specs.ts_returned_dim,
                          self.specs.ts_trashed_dim,
                          self.specs.ts_found_dim,
                          self.specs.ts_missing_dim ]

        for ( i, name, dim) in zip( cat_out_index, cat_out_names, cat_out_dims) :
            if dim is not None :
                layer_name = 'Output-' + str( 3 + i )
                layer = Dense( units = dim,
                               input_dim = self.layer_sizes[-1],
                               activation = 'softmax',
                               name = layer_name )
                TD_name = name + '_TS'
                Z_hat = TimeDistributed( layer,
                                         name = TD_name)( self.layer_outputs[-1] )
                self.learnable_layers.append( TD_name )
                self.outputs_list.append( Z_hat )

        self.model = Model( inputs = [ self.X_vecs, self.X_ts],
                            outputs = self.outputs_list )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def plot_model( self, filename = 'Cassian.png') :

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)

## =====================================================================================
#class CassianSimulator ( CassianModel ) :
#
#    def __init__( self, batch_size,
#                  vector_input_dim, timeseries_input_dim,
#                  vector_embedding_dim, layer_sizes ) :
#
#        super( CassianSimulator, self).__init__( batch_size, vec)
#
#        return