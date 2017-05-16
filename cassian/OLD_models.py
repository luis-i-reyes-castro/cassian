"""
@author: Luis I. Reyes Castro
"""

# =====================================================================================
import numpy as np
# -------------------------------------------------------------------------------------
from keras import backend as K
from keras import models as KM
from keras import layers as KL
from keras.layers.recurrent import GRU as KL_RNN
from keras.utils.visualize_util import plot
# -------------------------------------------------------------------------------------
#from cassian.batchers import Batcher

# =====================================================================================
class Cassian_Base :
    # ---------------------------------------------------------------------------------
    batch1_X = None
    batch1_Y = None
    batch2_X = None
    batch2_Y = None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, dataset, nb_products, nb_dates, stateful = False,
                  ff_layer_sizes = [ 128, 128], rec_layer_sizes = [ 128, 128] ) :
        # -----------------------------------------------------------------------------
        self.data_info        = dataset.info
        self.nb_products      = nb_products
        self.nb_dates         = nb_dates
        self.stateful         = stateful
        self.ff_layer_sizes   = ff_layer_sizes
        self.rec_layer_sizes  = rec_layer_sizes
        self.learnable_layers = []
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        t_shape = ( self.nb_products, self.data_info.dim_X_vecs)
        self.X_vecs = KL.Input( batch_shape = t_shape, name = 'X_vecs')
        self.X_vecs_seqs = KL.RepeatVector( self.nb_dates,
                                            name = 'Repeat-in-Time')( self.X_vecs)
        # -----------------------------------------------------------------------------
        t_shape = ( self.nb_products, self.nb_dates, self.data_info.dim_X_seqs)
        self.X_seqs = KL.Input( batch_shape = t_shape, name = 'X_seqs')
        # -----------------------------------------------------------------------------
        t_shape = ( self.nb_products, self.nb_dates, self.data_info.dim_X_date)
        self.X_date = KL.Input( batch_shape = t_shape, name = 'X_date')
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        input_seqs_A = KL.merge( [ self.X_vecs_seqs, self.X_date],
                                 mode = 'concat', concat_axis = -1,
                                 name = 'Input_seqs_A')
        input_seqs_B = KL.merge( [ self.X_seqs, self.X_date],
                                 mode = 'concat', concat_axis = -1,
                                 name = 'Input_seqs_B')
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        for ( layer, layer_size) in enumerate( self.ff_layer_sizes ) :
            # -------------------------------------------------------------------------
            layer_as_str   = str( layer + 1 )
            # -------------------------------------------------------------------------
            ff_layer_name = 'FF_layer-' + layer_as_str
            ff_layer = KL.Dense( activation = 'softsign',
                                 input_dim = input_seqs_A.get_shape()[-1].value,
                                 output_dim = layer_size, name = ff_layer_name)
            # -------------------------------------------------------------------------
            ff_vecs_name = 'Apply-in-time-' + layer_as_str
            ff_vecs = KL.TimeDistributed( layer = ff_layer,
                                          name = ff_vecs_name)( input_seqs_A )
            # -------------------------------------------------------------------------
            self.learnable_layers.append( ff_vecs_name )
            # -------------------------------------------------------------------------
            input_seqs_A = ff_vecs
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        input_seqs_AB = KL.merge( [ input_seqs_A, input_seqs_B],
                                  mode = 'concat', concat_axis = -1,
                                  name = 'Input_seqs_AB')
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        for ( layer, layer_size) in enumerate( self.rec_layer_sizes ) :
            # -------------------------------------------------------------------------
            layer_as_str   = str( layer + 1 )
            # -------------------------------------------------------------------------
            rec_seqs_name = 'REC_layer-' + layer_as_str
            rec_seqs = KL_RNN( activation = 'softsign',
                               inner_activation = 'sigmoid',
                               output_dim = layer_size,
                               return_sequences = True, stateful = self.stateful,
                               name = rec_seqs_name)( input_seqs_AB )
            # -------------------------------------------------------------------------
            self.learnable_layers.append( rec_seqs_name )
            # -------------------------------------------------------------------------
            input_seqs_AB = rec_seqs
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        self.Y_seqs_raw = rec_seqs
        self.Y_seqs     = None
        self.model      = None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def copy_weights( self, original_obj) :
        # -----------------------------------------------------------------------------
        for layer in self.learnable_layers :
            # -------------------------------------------------------------------------
            original_weights = original_obj.model.get_layer(layer).get_weights()
            # -------------------------------------------------------------------------
            self.model.get_layer(layer).set_weights( original_weights )
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def draw_model( self, filename = 'Cassian.png') :
        # -----------------------------------------------------------------------------
        plot( self.model, show_shapes = True, to_file = filename)
        # -----------------------------------------------------------------------------

# =====================================================================================
class Cassian_Simulator( Cassian_Base ) :
    # ---------------------------------------------------------------------------------
    batcher = None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, dataset, nb_products, nb_dates, stateful = False,
                  ff_layer_sizes = [ 128, 128], rec_layer_sizes = [ 128, 128] ) :
        # -----------------------------------------------------------------------------
        Cassian_Base.__init__( self, dataset, nb_products, nb_dates,
                               ff_layer_sizes, rec_layer_sizes)
        # -----------------------------------------------------------------------------
        output_layer_name = 'Output_layer'
        output_layer = KL.Dense( output_dim = 1,
                                 input_dim = self.Y_seqs_raw.get_shape()[-1].value,
                                 activation = lambda x : K.exp(x),
                                 name = output_layer_name)
        # -----------------------------------------------------------------------------
        output_name = 'Apply-in-Time'
        self.Y_seqs = KL.TimeDistributed( output_layer,
                                          name = output_name)( self.Y_seqs_raw )
        # -----------------------------------------------------------------------------
        self.learnable_layers.append( output_name )
        # -----------------------------------------------------------------------------
        self.model = KM.Model( input = [ self.X_vecs, self.X_seqs, self.X_date],
                               output = self.Y_seqs)
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def copy( self, nb_products, nb_dates, stateful = False) :
        # -----------------------------------------------------------------------------
        copy = Cassian_Simulator( self.data_info,
                                  self.nb_products, self.nb_dates, self.stateful,
                                  self.ff_layer_sizes, self.rec_layer_sizes)
        # -----------------------------------------------------------------------------
        copy.copy_weights(self)
        # -----------------------------------------------------------------------------
        return copy
#    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#    def setup_training( self, batch_specs, X_vecs, X_seqs, Y) :
#        # -----------------------------------------------------------------------------
#        self.model.compile( loss = 'poisson',
#                            metrics = [ 'mean_absolute_error' ],
#                            optimizer = 'adam' )
#        # -----------------------------------------------------------------------------
#        self.batcher = Batcher( batch_specs, X_vecs, X_seqs, Y)
#        # -----------------------------------------------------------------------------
#        empty_batch_X_vecs = \
#        np.zeros( ( self.nb_products, self.dim_X_vecs), dtype = 'float32')
#        empty_batch_X_seqs = \
#        np.zeros( ( self.nb_products, self.nb_dates, self.dim_X_seqs),
#                  dtype = 'float32')
#        # -----------------------------------------------------------------------------
#        self.batch1_X = [ empty_batch_X_vecs, empty_batch_X_seqs]
#        # -----------------------------------------------------------------------------
#        self.batch1_Y = \
#        np.zeros( ( self.nb_products, self.nb_dates, 1), dtype = 'float32')
#    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#    def train( self) :
#        # -----------------------------------------------------------------------------
#        reset_states_flag = \
#        self.batcher.query( self.batch1_X, self.batch1_Y)
#        # -----------------------------------------------------------------------------
#        loss_and_metrics = self.model.train_on_batch( self.batch1_X, self.batch1_Y)
#        # -----------------------------------------------------------------------------
#        if reset_states_flag :
#            self.model.reset_states()
#        # -----------------------------------------------------------------------------
#        return loss_and_metrics
#        # -----------------------------------------------------------------------------
