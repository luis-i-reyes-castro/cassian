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
from keras import activations, initializers, regularizers

# =====================================================================================
class NonlinearPID ( Layer ) :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, units,
                  activation = 'softsign',
                  stateful = False,
                  return_sequences = False,
                  go_backwards = False,
                  unroll = False,
                  mat_R_z_initializer = 'glorot_uniform',
                  mat_R_z_regularizer = None,
                  vec_b_z_initializer = 'zero',
                  vec_b_z_regularizer = None,
                  mat_R_0_initializer = 'glorot_uniform',
                  mat_R_0_regularizer = None,
                  vec_b_0_initializer = 'zero',
                  vec_b_0_regularizer = None,
                  mat_R_b_initializer = 'glorot_uniform',
                  mat_R_b_regularizer = None,
                  mat_W_p_initializer = 'glorot_uniform',
                  mat_W_p_regularizer = None,
                  mat_W_i_initializer = 'glorot_uniform',
                  mat_W_i_regularizer = None,
                  mat_W_d_initializer = 'glorot_uniform',
                  mat_W_d_regularizer = None,
                  activity_regulizer = None,
                  dropout_u = 0.,
                  dropout_x = 0., **kwargs ) :

        super( NonlinearPID, self).__init__(**kwargs)

        self.units            = units
        self.stateful         = stateful
        self.return_sequences = return_sequences
        self.go_backwards     = go_backwards
        self.unroll           = unroll
        self.supports_masking = True

        self.input_spec = [ InputSpec( ndim = 2), InputSpec( ndim = 3) ]
        self.state_spec = InputSpec( ndim = 2)

        self.activation = activations.get(activation)

        self.mat_R_z_initializer = initializers.get(mat_R_z_initializer)
        self.mat_R_z_regularizer = regularizers.get(mat_R_z_regularizer)
        self.vec_b_z_initializer = initializers.get(vec_b_z_initializer)
        self.vec_b_z_regularizer = regularizers.get(vec_b_z_regularizer)

        self.mat_R_0_initializer = initializers.get(mat_R_0_initializer)
        self.mat_R_0_regularizer = regularizers.get(mat_R_0_regularizer)
        self.vec_b_0_initializer = initializers.get(vec_b_0_initializer)
        self.vec_b_0_regularizer = regularizers.get(vec_b_0_regularizer)

        self.mat_R_b_initializer = initializers.get(mat_R_b_initializer)
        self.mat_R_b_regularizer = regularizers.get(mat_R_b_regularizer)

        self.mat_W_p_initializer = initializers.get(mat_W_p_initializer)
        self.mat_W_p_regularizer = regularizers.get(mat_W_p_regularizer)

        self.mat_W_i_initializer = initializers.get(mat_W_i_initializer)
        self.mat_W_i_regularizer = regularizers.get(mat_W_i_regularizer)
        self.mat_W_d_initializer = initializers.get(mat_W_d_initializer)
        self.mat_W_d_regularizer = regularizers.get(mat_W_d_regularizer)

        self.activity_regularizer = regularizers.get(activity_regulizer)

        self.dropout_u = min( 1., max( 0., dropout_u) )
        self.dropout_x = min( 1., max( 0., dropout_x) )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def check_inputs( self, inputs) :

        if not isinstance( inputs, list) or len(inputs) != 2 :
            raise ValueError( 'This layer should be called on a list of two inputs.' )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_output_shape( self, input_shape) :

        self.check_inputs( input_shape)

        U_input_shape = input_shape[0]
        X_input_shape = input_shape[1]
        batch_size    = U_input_shape[0]
        timesteps     = X_input_shape[1]

        if batch_size != X_input_shape[0] :
            raise ValueError( 'Batch size of vector and timeseries inputs ' +
                              'must match (i.e., must be the same).' )
        if batch_size is None :
            raise ValueError( 'This layer is learning an initial state and so ' +
                              'it needs to know its batch size.' )

        if self.return_sequences :
            return ( batch_size, timesteps, self.units)

        return ( batch_size, self.units)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def build( self, input_shape) :

        self.check_inputs( input_shape)

        U_input_shape = input_shape[0]
        X_input_shape = input_shape[1]

        batch_size = U_input_shape[0]
        timesteps  = X_input_shape[1]
        U_dim      = U_input_shape[1]
        X_dim      = X_input_shape[2]

        if batch_size is None :
            raise ValueError( 'This layer is learning an initial state and so ' +
                              'it needs to know its batch size.' )
        elif batch_size != X_input_shape[0] :
            raise ValueError( 'Batch size of static and dynamic inputs ' +
                              'must match (i.e., must be the same).' )

        self.input_spec = [ InputSpec( shape = ( batch_size, U_dim) ),
                            InputSpec( shape = ( batch_size, timesteps, X_dim) ) ]
        self.state_spec =   InputSpec( shape = ( batch_size, self.units) )

        self.mat_R_z = self.add_weight( name = 'mat_R_z',
                                        shape = ( U_dim, 3 * self.units),
                                        initializer = self.mat_R_z_initializer,
                                        regularizer = self.mat_R_z_regularizer)

        self.vec_b_z = self.add_weight( name = 'vec_b_z',
                                        shape = ( 1, 3 * self.units),
                                        initializer = self.vec_b_z_initializer,
                                        regularizer = self.vec_b_z_regularizer)
        self.vec_b_z = K.tile( self.vec_b_z, ( batch_size, 1) )

        self.mat_R_0 = self.add_weight( name = 'mat_R_0',
                                        shape = ( U_dim, self.units + X_dim),
                                        initializer = self.mat_R_0_initializer,
                                        regularizer = self.mat_R_0_regularizer)

        self.vec_b_0 = self.add_weight( name = 'vec_b_0',
                                        shape = ( 1, self.units + X_dim),
                                        initializer = self.vec_b_0_initializer,
                                        regularizer = self.vec_b_0_regularizer)
        self.vec_b_0 = K.tile( self.vec_b_0, ( batch_size, 1) )

        self.mat_R_b = self.add_weight( name = 'mat_R_b',
                                        shape = ( U_dim, self.units),
                                        initializer = self.mat_R_b_initializer,
                                        regularizer = self.mat_R_b_regularizer)

        self.mat_W_p = self.add_weight( name = 'mat_W_p',
                                        shape = ( X_dim, self.units),
                                        initializer = self.mat_W_p_initializer,
                                        regularizer = self.mat_W_p_regularizer)

        self.mat_W_i = self.add_weight( name = 'mat_W_i',
                                        shape = ( X_dim, self.units),
                                        initializer = self.mat_W_i_initializer,
                                        regularizer = self.mat_W_i_regularizer)

        self.mat_W_d = self.add_weight( name = 'mat_W_d',
                                        shape = ( X_dim, self.units),
                                        initializer = self.mat_W_d_initializer,
                                        regularizer = self.mat_W_d_regularizer)

        self.initial_i = K.zeros( ( batch_size, self.units) )
        self.initial_x = K.zeros( ( batch_size, X_dim) )

        if self.stateful :
            self.states = [ K.zeros_like(self.initial_i),
                            K.zeros_like(self.initial_x) ]
            self.reset_states()
        else :
            self.states = [ None, None ]

        super( NonlinearPID, self).build( input_shape)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __call__(self, inputs, **kwargs) :

        self.check_inputs( inputs)

        return super( NonlinearPID, self).__call__( inputs, **kwargs)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def call( self, inputs, mask = None, training = None) :

        # -----------------------------------------------------------------------------
        self.check_inputs( inputs)

        U_vecs = inputs[0] # shape = ( batch_size, U_dim)
        X_vecs = inputs[1] # shape = ( batch_size, timesteps, X_dim)

        if K.int_shape( U_vecs)[0] != K.int_shape( X_vecs)[0] :
            raise ValueError( 'Batch size of static and dynamic inputs ' +
                              'must match (i.e., must be the same).' )
        if self.unroll and K.int_shape( X_vecs)[1] is None :
            raise ValueError( 'Cannot use option unroll if number of timesteps ' +
                              'is undefined.' )

        # Applies dropout to the U-inputs if applicable. Regardless, at the end
        # tensor U_vecs has shape ( batch_size, U_dim).
        if 0 < self.dropout_u < 1 :
            U_ones = K.ones_like( U_vecs)
            U_mask = K.dropout( U_ones, self.dropout_u)
            U_vecs._uses_learning_phase = True
            U_vecs = K.in_train_phase( U_vecs * U_mask, U_vecs, training)

        # Computes U-input-dependent biases
        U_bias = K.dot( U_vecs, self.mat_R_b) # shape = ( batch_size, units)

        # Computes channel selectors (i.e. Z-vectors) from U-inputs, resulting in
        # tensors Z_p, Z_i and Z_d, each with shape ( batch_size, units).
        Z_pid = K.dot( U_vecs, self.mat_R_z) + self.vec_b_z
        # shape = ( batch_size, 3 * units)
        Z_pid = K.reshape( Z_pid, ( -1, self.units, 3) )
        # shape = ( batch_size, units, 3)
        Z_p = Z_pid[ :, :, 0]
        Z_i = Z_pid[ :, :, 1]
        Z_d = Z_pid[ :, :, 2]

        # Builds X-inputs dropout mask if applicable
        if 0 < self.dropout_x < 1 :
            X_ones = K.ones_like( X_vecs[ :, 0, :] ) # shp = ( batch, X_dim)
            X_mask = K.dropout( X_ones, self.dropout_x)
            X_mask = K.in_train_phase( X_mask, X_ones, training)

        # Defines list of initial states. At the end list_of_initial_states
        # contains two tensors: the first is the initial integral term with
        # shape ( batch_size, units); the second is the previous input with
        # shape ( batch_size, X_dim).
        if self.stateful :
            list_of_initial_states = self.states
        else :
            initial_ix = K.dot( U_vecs, self.mat_R_0) + self.vec_b_0
            # shape = ( batch_size, units + X_dim)
            self.initial_i += initial_ix[ :, : self.units ] # shp = ( batch, units)
            self.initial_x += initial_ix[ :, self.units : ] # shp = ( batch, X_dim)
            list_of_initial_states = [ self.initial_i, self.initial_x ]

        # Applies dropout to the initial X-input if applicable
        if 0 < self.dropout_x < 1 :
            list_of_initial_states[1] *= X_mask

        # -----------------------------------------------------------------------------
        # For each timestep t ...
        def recursion( X_t, list_of_previous_states) :

            # Input X_t is a tensor of shape ( batch_size, X_dim)
            if self.dropout_x :
                X_t = X_t * X_mask
            # List of previous states contains two tensors:
            # Previous integral I_{t-1} with shape = ( batch_size, units);
            # Previous input X_{t-1} with shape = ( batch_size, X_dim).
            I_tm1 = list_of_previous_states[0]
            X_tm1 = list_of_previous_states[1]

            # Computes (P) proportional, (I) integral and (D) difference terms.
            # Each of them has shape ( batch_size, units).
            P_t = self.activation( U_bias + K.dot( X_t, self.mat_W_p) )
            I_t = self.activation( I_tm1 + K.dot( X_t, self.mat_W_i) )
            D_t = self.activation( K.dot( X_t - X_tm1, self.mat_W_d) )

            # Computes output tensor, which has shape ( batch_size, units).
            Y_t = ( Z_p * P_t ) + ( Z_i * I_t ) + ( Z_d * D_t )
            if 0 < self.dropout_u + self.dropout_x :
                Y_t._uses_learning_phase = True

            # List of current states contains two tensors:
            # Current integral I_t with shape = ( batch_size, units);
            # Current input X_t with shape = ( batch_size, X_dim).
            list_of_states = [ I_t, X_t ]

            return ( Y_t, list_of_states)

        # -----------------------------------------------------------------------------
        last_output, outputs, states = K.rnn( step_function = recursion,
                                              inputs = X_vecs,
                                              initial_states = list_of_initial_states,
                                              mask = mask,
                                              go_backwards = self.go_backwards,
                                              unroll = self.unroll)

        if self.stateful :
            updates = [ ( self.states[0], states[0]) ]
            self.add_update( updates, inputs)

        if 0 < self.dropout_u + self.dropout_x :
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences :
            return outputs

        return last_output

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_and_set_initial_states( self, U_vecs) :

        if not self.stateful :
            raise AttributeError( 'Layer must be stateful.' )

        if not isinstance( U_vecs, np.ndarray) :
            raise ValueError( 'Parameter U_vecs must be a numpy array.' )

        batch_size    = self.input_spec[0].shape[0]
        U_dim         = self.input_spec[0].shape[1]
        correct_shape = ( batch_size, U_dim)

        if U_vecs.shape != correct_shape :
            raise ValueError( 'Expected U_vecs parameter to have shape ' +
                              str(correct_shape) + ' but got passed an array ' +
                              'of shape ' + str( U_vecs.shape ) + '.' )

        initial_ix = np.dot( U_vecs, K.get_value(self.mat_R_0) ) \
                   + K.get_value( self.vec_b_0)

        K.set_value( self.initial_i, initial_ix[ :, : self.units ] )
        K.set_value( self.initial_x, initial_ix[ :, self.units : ] )

        self.reset_states()

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def reset_states( self) :

        if not self.stateful :
            raise AttributeError( 'Layer must be stateful.' )

        K.set_value( self.states[0], K.get_value(self.initial_i) )
        K.set_value( self.states[1], K.get_value(self.initial_x) )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_mask( self, inputs, mask):

        if self.return_sequences:
            return mask

        return None

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_config(self) :

        base_config = super( NonlinearPID, self).get_config()

        config = { 'units' : self.units,
                   'activation' : self.activation,
                   'stateful': self.stateful,
                   'return_sequences' : self.return_sequences,
                   'go_backwards' : self.go_backwards,
                   'unroll': self.unroll,
                   'dropout_u' : self.dropout_u,
                   'dropout_x' : self.dropout_x }

        return dict( list( base_config.items() ) + list( config.items() ) )
