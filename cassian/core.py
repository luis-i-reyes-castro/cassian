"""
@author: Luis I. Reyes-Castro

COPYRIGHT

All contributions by Luis I. Reyes-Castro:
Copyright (c) 2017, Luis Ignacio Reyes Castro.
All rights reserved.

LICENSE

The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

# =====================================================================================
class HybridUnit ( Layer ) :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, units,
                  stateful = False,
                  return_sequences = False,
                  go_backwards = False,
                  unroll = False,
                  mat_A_s_initializer = 'glorot_uniform',
                  vec_b_s_initializer = 'zero',
                  mat_A_0_initializer = 'glorot_uniform',
                  vec_b_0_initializer = 'zero',
                  mat_W_p_initializer = 'glorot_uniform',
                  vec_b_p_initializer = 'zero',
                  mat_W_i_initializer = 'glorot_uniform',
                  vec_b_i_initializer = 'zero',
                  mat_W_d_initializer = 'glorot_uniform',
                  vec_b_d_initializer = 'zero',
                  activity_reg = None,
                  vector_dropout = 0., input_dropout = 0., **kwargs) :

        super( HybridUnit, self).__init__(**kwargs)

        self.units            = units
        self.stateful         = stateful
        self.return_sequences = return_sequences
        self.go_backwards     = go_backwards
        self.unroll           = unroll
        self.supports_masking = True

        self.input_spec = [ InputSpec( ndim = 2), InputSpec( ndim = 3) ]
        self.state_spec = InputSpec( ndim = 2)

        self.main_activation = lambda x : K.softsign(x)
        self.gate_activation = lambda x : 0.5 + 0.5 * K.softsign( 2.0 * x )

        self.mat_A_s_initializer = initializers.get(mat_A_s_initializer)
        self.vec_b_s_initializer = initializers.get(vec_b_s_initializer)
        self.mat_A_0_initializer = initializers.get(mat_A_0_initializer)
        self.vec_b_0_initializer = initializers.get(vec_b_0_initializer)

        self.mat_W_p_initializer = initializers.get(mat_W_p_initializer)
        self.vec_b_p_initializer = initializers.get(vec_b_p_initializer)
        self.mat_W_i_initializer = initializers.get(mat_W_i_initializer)
        self.vec_b_i_initializer = initializers.get(vec_b_i_initializer)
        self.mat_W_d_initializer = initializers.get(mat_W_d_initializer)
        self.vec_b_d_initializer = initializers.get(vec_b_d_initializer)

        self.activity_regularizer = regularizers.get( activity_reg)

        self.vector_dropout = min( 1., max( 0., vector_dropout) )
        self.input_dropout  = min( 1., max( 0., input_dropout) )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def check_inputs( self, inputs) :

        if not isinstance( inputs, list) or len(inputs) != 2 :
            raise ValueError( 'This layer should be called on a list of two inputs.' )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_output_shape( self, input_shape) :

        self.check_inputs( input_shape)

        vector_input_shape     = input_shape[0]
        batch_size             = vector_input_shape[0]
        timeseries_input_shape = input_shape[1]
        timesteps              = timeseries_input_shape[1]

        if batch_size is None :
            raise ValueError( 'This layer is learning an initial state and so ' +
                              'it needs to know its batch size.' )
        if vector_input_shape[0] != timeseries_input_shape[0] :
            raise ValueError( 'Batch size of vector and timeseries inputs ' +
                              'must match (i.e., must be the same).' )

        if self.return_sequences :
            return ( batch_size, timesteps, self.units)

        return ( batch_size, self.units)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def build( self, input_shape) :

        self.check_inputs( input_shape)

        vec_input_shape = input_shape[0]
        batch_size      = vec_input_shape[0]
        vec_input_dim   = vec_input_shape[1]

        ts_input_shape  = input_shape[1]
        timesteps       = ts_input_shape[1]
        ts_input_dim    = ts_input_shape[2]

        if batch_size is None :
            raise ValueError( 'This layer is learning an initial state and so ' +
                              'it needs to know its batch size.' )
        elif batch_size != ts_input_shape[0] :
            raise ValueError( 'Batch size of vector and timeseries inputs ' +
                              'must match (i.e., must be the same).' )

        self.input_spec = [ InputSpec( shape = ( batch_size, vec_input_dim) ),
                            InputSpec( shape = ( batch_size, timesteps, ts_input_dim) ) ]
        self.state_spec =   InputSpec( shape = ( batch_size, self.units) )

        mat_A_s_shape = ( vec_input_dim, self.units)
        vec_b_s_shape = ( 1, self.units)
        mat_A_0_shape = ( vec_input_dim, self.units)
        vec_b_0_shape = ( 1, self.units)

        self.mat_A_s = self.add_weight( name = 'mat_A_s',
                                        shape = mat_A_s_shape,
                                        initializer = self.mat_A_s_initializer)

        self.vec_b_s = self.add_weight( name = 'vec_b_s',
                                        shape = vec_b_s_shape,
                                        initializer = self.vec_b_s_initializer)
        self.vec_b_s = K.tile( self.vec_b_s, ( batch_size, 1) )

        self.mat_A_0 = self.add_weight( name = 'mat_A_0',
                                        shape = mat_A_0_shape,
                                        initializer = self.mat_A_0_initializer)

        self.vec_b_0 = self.add_weight( name = 'vec_b_0',
                                        shape = vec_b_0_shape,
                                        initializer = self.vec_b_0_initializer)
        self.vec_b_0 = K.tile( self.vec_b_0, ( batch_size, 1) )

        mat_W_p_shape = ( ts_input_dim, self.units)
        vec_b_p_shape = ( 1, self.units)
        mat_W_i_shape = ( ts_input_dim, self.units)
        vec_b_i_shape = ( 1, self.units)
        mat_W_d_shape = ( ts_input_dim, self.units)
        vec_b_d_shape = ( 1, self.units)

        self.mat_W_p = self.add_weight( name = 'mat_W_p',
                                        shape = mat_W_p_shape,
                                        initializer = self.mat_W_p_initializer)

        self.vec_b_p = self.add_weight( name = 'vec_b_p',
                                        shape = vec_b_p_shape,
                                        initializer = self.vec_b_p_initializer)
        self.vec_b_p = K.tile( self.vec_b_p, ( batch_size, 1) )

        self.mat_W_i = self.add_weight( name = 'mat_W_i',
                                        shape = mat_W_i_shape,
                                        initializer = self.mat_W_i_initializer)

        self.vec_b_i = self.add_weight( name = 'vec_b_i',
                                        shape = vec_b_i_shape,
                                        initializer = self.vec_b_i_initializer)
        self.vec_b_i = K.tile( self.vec_b_i, ( batch_size, 1) )

        self.mat_W_d = self.add_weight( name = 'mat_W_d',
                                        shape = mat_W_d_shape,
                                        initializer = self.mat_W_d_initializer)

        self.vec_b_d = self.add_weight( name = 'vec_b_d',
                                        shape = vec_b_d_shape,
                                        initializer = self.vec_b_d_initializer)
        self.vec_b_d = K.tile( self.vec_b_d, ( batch_size, 1) )

        initial_state_shape = ( batch_size, self.units)
        self.initial_state  = K.zeros( initial_state_shape)

        if self.stateful :
            self.states = [ K.zeros( initial_state_shape) ]
            self.reset_states()
        else :
            self.states = [ None ]

        super( HybridUnit, self).build( input_shape)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __call__(self, inputs, **kwargs) :

        self.check_inputs( inputs)

        return super( HybridUnit, self).__call__( inputs, **kwargs)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def call( self, inputs, mask = None, training = None) :

        # -----------------------------------------------------------------------------
        self.check_inputs( inputs)

        X_vecs = inputs[0] # shape = ( batch_size, vec_input_dim)
        X_ts   = inputs[1] # shape = ( batch_size, timesteps, ts_input_dim)

        if K.int_shape( X_vecs)[0] != K.int_shape( X_ts)[0] :
            raise ValueError( 'Batch size of vector and timeseries inputs ' +
                              'must match (i.e., must be the same).' )
        if self.unroll and K.int_shape( X_ts)[1] is None :
            raise ValueError( 'Cannot unroll the RNN if the number of timesteps ' +
                              'is undefined.' )

        # Applies dropout to the vector inputs if applicable. Regardless, at the end
        # tensor X_vecs has shape ( batch_size, vec_input_dim).
        if 0 < self.vector_dropout < 1 :
            vec_input_ones = K.ones_like( X_vecs)
            array_01s = K.dropout( vec_input_ones, self.vector_dropout)
            X_vecs._uses_learning_phase = True
            X_vecs = K.in_train_phase( X_vecs * array_01s, X_vecs, training)

        # Defines list of initial states. At the end H_0 is a list
        # containing a single tensor of shape ( batch_size, units).
        if self.stateful :
            H_0 = self.states
        else :
            batch_size = K.shape(X_ts)[0]
            self.initial_state += \
            self.main_activation( K.dot( X_vecs, self.mat_W_0) \
                                + K.tile( self.vec_b_0, ( batch_size, 1) ) )
            H_0 = [ self.initial_state ]

        # Builds input dropout mask for if applicable. Regardless, at the end
        # tensor State_dp_mask has shape ( batch_size, ts_input_dim).
        input_ones    = K.ones_like( X_ts[ :, 0, :] )
        Input_dp_mask = input_ones
        if 0 < self.input_dropout < 1 :
            array_01s = K.dropout( input_ones, self.input_dropout)
            Input_dp_mask = K.in_train_phase( array_01s, input_ones, training)

        # Precomputes feedforward terms
        X_vecs_dot_mat_P_d = K.dot( X_vecs, self.mat_P_d)

        # -----------------------------------------------------------------------------
        # For each timestep t ...
        def rnn_recursion( X_t, list_of_H_tm1) :

            # Input X_t is a tensor of shape ( batch_size, ts_input_dim).
            X_t = X_t * Input_dp_mask
            # Previous state H_{t-1} is list containing a single tensor of
            # shape ( batch_size, units).
            H_tm1 = list_of_H_tm1[0]

            # Computes state update vectors; shape ( batch_size, units).
            H_t = self.main_activation( H_tm1 \
                                      + X_vecs_dot_mat_P_d
                                      + K.dot( X_t, self.mat_W_d) \
                                      + K.tile( self.vec_b_d, ( batch_size, 1) ) )

            if 0 < self.input_dropout :
                H_t._uses_learning_phase = True

            return ( H_t, [H_t] )

        # -----------------------------------------------------------------------------
        last_output, outputs, states = K.rnn( step_function = rnn_recursion,
                                              inputs = X_ts,
                                              initial_states = H_0,
                                              mask = mask,
                                              go_backwards = self.go_backwards,
                                              unroll = self.unroll)

        if self.stateful :
            updates = [ ( self.states[0], states[0]) ]
            self.add_update( updates, inputs)

        if 0 < self.input_dropout :
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences :
            return outputs

        return last_output

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_and_set_initial_states( self, X_vectors) :

        if not self.stateful :
            raise AttributeError( 'Layer must be stateful.' )

        if not isinstance( X_vectors, np.ndarray) :
            raise ValueError( 'Parameter X_vectors must be a numpy array.' )

        batch_size       = self.input_spec[0].shape[0]
        vector_input_dim = self.input_spec[0].shape[1]

        if X_vectors.shape != ( batch_size, vector_input_dim) :
            raise ValueError( 'Expected X_vectors parameter to have shape ' +
                              str( ( batch_size, vector_input_dim) ) +
                              ' but got passed an array of shape ' +
                              str( X_vectors.shape ) + '.' )

        H_0 = np.dot( X_vectors, K.get_value( self.mat_W_0) ) \
            + np.tile( K.get_value( self.vec_b_0), ( batch_size, 1) )
        H_0 = H_0 / ( 1 + np.abs( H_0 ) )

        K.set_value( self.initial_state, H_0)
        self.reset_states()

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def reset_states( self) :

        if not self.stateful :
            raise AttributeError( 'Layer must be stateful.' )

        K.set_value( self.states[0], K.get_value( self.initial_state) )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_mask( self, inputs, mask):

        if self.return_sequences:
            return mask

        return None

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_config(self) :

        base_config = super( HybridUnit, self).get_config()

        config = { 'units' : self.units,
                   'stateful': self.stateful,
                   'return_sequences' : self.return_sequences,
                   'go_backwards' : self.go_backwards,
                   'unroll': self.unroll,
                   'vector_dropout' : self.vector_dropout,
                   'input_dropout' : self.input_dropout }

        return dict( list( base_config.items() ) + list( config.items() ) )
