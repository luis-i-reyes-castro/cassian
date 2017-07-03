"""
@author: Luis I. Reyes-Castro

COPYRIGHT

All contributions by Luis I. Reyes-Castro:
Copyright (c) 2017, Luis Ignacio Reyes Castro.
All rights reserved.
"""

import numpy as np

class BatchSpecifications :

    batch_size         = 0
    timesteps          = 0
    vec_dim            = 0
    ts_dim             = 0
    ts_replenished_dim = 0
    ts_returned_dim    = 0
    ts_trashed_dim     = 0
    ts_found_dim       = 0
    ts_missing_dim     = 0

class BatchSample :

    size          = 0
    X_vec         = np.array([])
    X_ts          = np.array([])
    Y_sold        = np.array([])
    Y_is_on_sale  = np.array([])
    Z_replenished = np.array([])
    Z_returned    = np.array([])
    Z_trashed     = np.array([])
    Z_found       = np.array([])
    Z_missing     = np.array([])

    def __init__( self, batch_specs = BatchSpecifications()) :

        self.size  = batch_specs.batch_size

        self.X_vec = np.zeros( ( batch_specs.batch_size,
                                 batch_specs.vec_dim),
                                 dtype = 'float32')
        self.X_ts = np.zeros( ( batch_specs.batch_size,
                                batch_specs.timesteps, batch_specs.ts_dim),
                                dtype = 'float32')

        empty_output_tensor = np.zeros( ( batch_specs.batch_size,
                                          batch_specs.timesteps, 1) )

        empty_output_tensor = empty_output_tensor.astype('float32')
        self.Y_sold         = empty_output_tensor.copy()
        self.Y_is_on_sale   = empty_output_tensor.copy()

        empty_output_tensor = empty_output_tensor.astype('int32')
        self.Z_replenished  = empty_output_tensor.copy()
        self.Z_returned     = empty_output_tensor.copy()
        self.Z_trashed      = empty_output_tensor.copy()
        self.Z_found        = empty_output_tensor.copy()
        self.Z_missing      = empty_output_tensor.copy()

        self.inputs  = [ self.X_vec, self.X_ts ]
        self.targets = [ self.Y_sold, self.Y_is_on_sale,
                         self.Z_replenished, self.Z_returned, self.Z_trashed,
                         self.Z_found, self.Z_missing ]

        return
