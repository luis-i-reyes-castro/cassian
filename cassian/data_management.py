"""
@author: Luis I. Reyes Castro
"""

# =====================================================================================
import copy as cp
import numpy as np
import pandas as pd
# -------------------------------------------------------------------------------------
from .db_clients import DIR_RESULT_SET
from .db_clients import RESULT_SET
from .convenience import serialize, de_serialize
from sklearn.model_selection import train_test_split
# -------------------------------------------------------------------------------------
INPUT_DIR = DIR_RESULT_SET
INPUT_FILE = RESULT_SET
OUTPUT_FILE = 'ready.pkl'

# =====================================================================================
class Dataset :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    store_id = 0
    # ---------------------------------------------------------------------------------
    info_main = pd.DataFrame()
    info_replenish = pd.DataFrame()
    info_other = pd.DataFrame()
    info_description = pd.DataFrame()
    # ---------------------------------------------------------------------------------
    num_skus = 0
    list_of_skus = []
    # ---------------------------------------------------------------------------------
    cat_replenished = None
    cat_returned = None
    cat_trashed = None
    cat_found = None
    cat_missing = None
    # ---------------------------------------------------------------------------------
    vectors = pd.DataFrame()
    categorizer = {}
    data = {}
    # ---------------------------------------------------------------------------------
    num_timesteps = 0
    vec_dim = 0
    ts_dim = 0
    ts_replenished_dim = 0
    ts_returned_dim = 0
    ts_trashed_dim = 0
    ts_found_dim = 0
    ts_missing_dim = 0
    list_of_sku_probs = []
    # ---------------------------------------------------------------------------------
    vec_mean = np.array([])
    vec_std = np.array([])
    ts_mean = np.array([])
    ts_std = np.array([])
    # ---------------------------------------------------------------------------------
    batch_specs = None
    batch_sample = None

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, store_id = 0) :

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        if store_id == 0 :
            return
        # -----------------------------------------------------------------------------
        self.store_id = store_id

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        filename = INPUT_DIR.replace( '[STORE-ID]', str(self.store_id))
        filename += INPUT_FILE
        data_raw = de_serialize( filename)
        # -----------------------------------------------------------------------------
        self.info_main = data_raw['info-main']
        self.info_replenish = data_raw['info-replenish']
        self.info_other = data_raw['info-other']
        self.info_description = data_raw['info-description']
        self.num_skus = len( self.info_main )
        self.list_of_skus = self.info_main.index.tolist()

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.vectors = pd.get_dummies( self.info_main)

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.categorizer = {}
        self.categorizer['replenished'] = \
        Timeseries_Categorizer( max_val = self.info_other['REPLENISHED_MAX'].max() )
        self.categorizer['returned'] = \
        Timeseries_Categorizer( max_val = self.info_other['RETURNED_MAX'].max() )
        self.categorizer['trashed'] = \
        Timeseries_Categorizer( max_val = self.info_other['TRASHED_MAX'].max() )
        self.categorizer['found'] = \
        Timeseries_Categorizer( max_val = self.info_other['FOUND_MAX'].max() )
        self.categorizer['missing'] = \
        Timeseries_Categorizer( max_val = self.info_other['MISSING_MAX'].max() )
        # -----------------------------------------------------------------------------
        self.data = {}
        # -----------------------------------------------------------------------------
        for sku in self.list_of_skus :
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            vec_ = self.vectors.loc[sku]
            ts_ = data_raw['timeseries'][ sku]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            self.data[sku] = SKU_Data( vector = vec_, timeseries = ts_)
            self.data[sku].categorize_ts( self.categorizer)

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.update_num_timesteps_and_list_of_sku_probs()
        # -----------------------------------------------------------------------------
        arbitrary_sku_obj = self.data[ self.list_of_skus[0] ]
        self.vec_dim = arbitrary_sku_obj.vec_dim
        self.ts_dim = arbitrary_sku_obj.ts_dim
        self.ts_replenished_dim = arbitrary_sku_obj.ts_replenished_dim
        self.ts_returned_dim = arbitrary_sku_obj.ts_returned_dim
        self.ts_trashed_dim = arbitrary_sku_obj.ts_trashed_dim
        self.ts_found_dim = arbitrary_sku_obj.ts_found_dim
        self.ts_missing_dim = arbitrary_sku_obj.ts_missing_dim

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.compute_mean_std()
        self.normalize_timeseries()

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def save( self) :
        filename = INPUT_DIR.replace( '[STORE-ID]', str(self.store_id))
        filename += OUTPUT_FILE
        serialize( self, filename)
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def load( store_id) :
        filename = INPUT_DIR.replace( '[STORE-ID]', str(store_id))
        filename += OUTPUT_FILE
        return de_serialize( filename)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def update_num_timesteps_and_list_of_sku_probs( self) :
        # -----------------------------------------------------------------------------
        self.num_timesteps = 0
        self.list_of_sku_probs = [ 0.0 for sku in self.list_of_skus ]
        # -----------------------------------------------------------------------------
        for sku in self.list_of_skus :
            self.num_timesteps += self.data[sku].num_timesteps
        # -----------------------------------------------------------------------------
        for ( i, sku) in enumerate( self.list_of_skus ) :
            self.list_of_sku_probs[i] = \
            float( self.data[sku].num_timesteps) / self.num_timesteps
        # -----------------------------------------------------------------------------
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_mean_std( self) :

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.vec_mean = np.zeros( shape = ( self.vec_dim, ), dtype = 'float32')
        self.vec_std = np.zeros( shape = ( self.vec_dim, ), dtype = 'float32')
        # -----------------------------------------------------------------------------
        self.vec_mean += self.vectors.mean( axis = 0).as_matrix()
        self.vec_std += self.vectors.std( axis = 0).as_matrix()

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.ts_mean = np.zeros( shape = ( self.ts_dim, ), dtype = 'float32')
        self.ts_std = np.zeros( shape = ( self.ts_dim, ), dtype = 'float32')
        # -----------------------------------------------------------------------------
        for ( i, sku) in enumerate( self.list_of_skus ) :
            self.ts_mean += self.list_of_sku_probs[i] \
                          * self.data[sku].get_ts_mean()
        # -----------------------------------------------------------------------------
        for ( i, sku) in enumerate( self.list_of_skus ) :
            self.ts_std += self.list_of_sku_probs[i] \
                         * self.data[sku].get_ts_mqd_about( self.ts_mean)
        # -----------------------------------------------------------------------------
        self.ts_std = np.sqrt( self.ts_std )

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def normalize_timeseries( self, de_normalize = False) :
        # -----------------------------------------------------------------------------
        for sku in self.list_of_skus :
            self.data[sku].normalize_ts( self.ts_mean, self.ts_std, de_normalize)
        # -----------------------------------------------------------------------------
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def split( self, train_fraction = 0.8):

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        list_of_skus_A, list_of_skus_B = \
        train_test_split( self.list_of_skus, train_size = train_fraction)

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        lists_of_skus = [ list_of_skus_A, list_of_skus_B]
        dataset_objects = []

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        for list_of_skus in lists_of_skus :
            # -------------------------------------------------------------------------
            ds_obj = Dataset()
            ds_obj.store_id = self.store_id
            # -------------------------------------------------------------------------
            good_rows = self.info_main.index.isin( list_of_skus)
            # -------------------------------------------------------------------------
            ds_obj.info_main = self.info_main.loc[ good_rows]
            ds_obj.info_replenish = self.info_replenish.loc[ good_rows]
            ds_obj.info_other = self.info_other.loc[ good_rows]
            ds_obj.info_description = self.info_description.loc[ good_rows]
            ds_obj.num_skus = len( list_of_skus)
            ds_obj.list_of_skus = list_of_skus.copy()
            # -------------------------------------------------------------------------
            ds_obj.vectors = self.vectors.loc[ good_rows]
            # -------------------------------------------------------------------------
            ds_obj.categorizer = {}
            for key in self.categorizer.keys() :
                ds_obj.categorizer[key] = self.categorizer[key].get_copy()
            # -------------------------------------------------------------------------
            ds_obj.data = {}
            for sku in ds_obj.list_of_skus :
                ds_obj.data[sku] = self.data[sku].get_copy()
            # -------------------------------------------------------------------------
            ds_obj.update_num_timesteps_and_list_of_sku_probs()
            # -------------------------------------------------------------------------
            ds_obj.vec_dim = self.vec_dim
            ds_obj.ts_dim = self.ts_dim
            ds_obj.ts_replenished_dim = self.ts_replenished_dim
            ds_obj.ts_returned_dim = self.ts_returned_dim
            ds_obj.ts_trashed_dim = self.ts_trashed_dim
            ds_obj.ts_found_dim = self.ts_found_dim
            ds_obj.ts_missing_dim = self.ts_missing_dim
            # -------------------------------------------------------------------------
            ds_obj.compute_mean_std()
            ds_obj.normalize_timeseries()
            # -------------------------------------------------------------------------
            dataset_objects.append( ds_obj)

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        return ( dataset_objects[0], dataset_objects[1])

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def setup_sampler( self, nb_samples, timesteps) :
        # -----------------------------------------------------------------------------
        self.batch_specs = Batch_Specs()
        self.batch_specs.nb_samples = nb_samples
        self.batch_specs.timesteps = timesteps
        self.batch_specs.vec_dim = self.vec_dim
        self.batch_specs.ts_dim = self.ts_dim
        # -----------------------------------------------------------------------------
        self.batch_sample = Batch_Sample( self.batch_specs)
        # -----------------------------------------------------------------------------
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def sample_batch( self) :
        # -----------------------------------------------------------------------------
        batch_skus = np.random.choice( a = self.list_of_skus,
                                       p = self.list_of_sku_probs,
                                       size = self.batch_specs.nb_samples)
        # -----------------------------------------------------------------------------
        for ( i, sku) in enumerate( batch_skus ) :
            self.data[ sku].sample( sample_obj = self.batch_sample.sample_obj,
                                    timesteps = self.batch_specs.timesteps)
            self.batch_sample.copy_sample_obj_into_pos(i)
        # -----------------------------------------------------------------------------
        return

# =====================================================================================
class SKU_Data :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    vector = pd.Series()
    vec = np.array([])
    vec_dim = 0
    timeseries = pd.DataFrame()
    num_timesteps = 0

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    ts = np.array([])
    ts_dim = 0
    # ---------------------------------------------------------------------------------
    ts_sold = np.array([])
    ts_is_on_sale = np.array([])
    ts_replenished = np.array([])
    ts_returned = np.array([])
    ts_trashed = np.array([])
    ts_found = np.array([])
    ts_missing = np.array([])
    # ---------------------------------------------------------------------------------
    ts_replenished_cat = np.array([])
    ts_replenished_dim = 0
    ts_returned_cat = np.array([])
    ts_returned_dim = 0
    ts_trashed_cat = np.array([])
    ts_trashed_dim = 0
    ts_found_cat = np.array([])
    ts_found_dim = 0
    ts_missing_cat = np.array([])
    ts_missing_dim = 0

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, vector, timeseries) :
        # -----------------------------------------------------------------------------
        self.vector = vector
        self.vec = self.vector.as_matrix().astype('float32')
        self.vec_dim = self.vec.shape[0]
        # -----------------------------------------------------------------------------
        self.timeseries = timeseries
        self.num_timesteps = len( self.timeseries ) - 1
        self.setup_ts()
        # -----------------------------------------------------------------------------
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def setup_ts( self) :
        # -----------------------------------------------------------------------------
        ts_cols = self.timeseries.columns.tolist()
        ts_cols.remove( 'UNIT_PRICE' )
        ts_cols.remove( 'REPLENISHED_FLAG' )
        ts_cols.remove( 'RETURNED_FLAG' )
        ts_cols.remove( 'TRASHED_FLAG' )
        ts_cols.remove( 'FOUND_FLAG' )
        ts_cols.remove( 'MISSING_FLAG' )
        # -----------------------------------------------------------------------------
        self.ts = self.timeseries[ ts_cols ].as_matrix().astype('float32')
        self.ts_dim = self.ts.shape[1]
        # -----------------------------------------------------------------------------
        self.ts_sold = \
        self.timeseries[ ['SOLD'] ].as_matrix().astype('float32')
        self.ts_is_on_sale = \
        self.timeseries[ ['IS_ON_SALE'] ].as_matrix().astype('float32')
        # -----------------------------------------------------------------------------
        self.ts_replenished = self.timeseries[ ['REPLENISHED'] ].as_matrix()
        self.ts_returned = self.timeseries[ ['RETURNED'] ].as_matrix()
        self.ts_trashed = self.timeseries[ ['TRASHED'] ].as_matrix()
        self.ts_found = self.timeseries[ ['FOUND'] ].as_matrix()
        self.ts_missing = self.timeseries[ ['MISSING'] ].as_matrix()
        # -----------------------------------------------------------------------------
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def categorize_ts( self, categorizers) :
        # -----------------------------------------------------------------------------
        self.ts_replenished_cat = \
        categorizers['replenished'].categorize( self.ts_replenished )
        self.ts_replenished_dim = categorizers['replenished'].num_categories
        # -----------------------------------------------------------------------------
        self.ts_returned_cat = \
        categorizers['returned'].categorize( self.ts_returned )
        self.ts_returned_dim = categorizers['replenished'].num_categories
        # -----------------------------------------------------------------------------
        self.ts_trashed_cat = \
        categorizers['trashed'].categorize( self.ts_trashed )
        self.ts_trashed_dim = categorizers['trashed'].num_categories
        # -----------------------------------------------------------------------------
        self.ts_found_cat = \
        categorizers['found'].categorize( self.ts_found )
        self.ts_found_dim = categorizers['found'].num_categories
        # -----------------------------------------------------------------------------
        self.ts_missing_cat = \
        categorizers['missing'].categorize( self.ts_missing )
        self.ts_missing_dim = categorizers['missing'].num_categories
        # -----------------------------------------------------------------------------
        return

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_copy( self) :
        return cp.deepcopy(self)

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_ts_mean( self) :
        return np.mean( self.ts[:-1,:], axis = 0)

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_ts_mqd_about( self, vector) :
        diff = self.ts[:-1,:] - vector
        return np.mean( np.square(diff), axis = 0)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def normalize_ts( self, ts_mean, ts_std, de_normalize = False) :
        if de_normalize is False :
            self.ts -= ts_mean
            self.ts /= ts_std
        else :
            self.ts *= ts_std
            self.ts += ts_mean
        return

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def sample( self, sample_obj, timesteps) :
        # -----------------------------------------------------------------------------
        i_start = np.random.randint( self.num_timesteps - timesteps )
        i_end = i_start + timesteps
        # -----------------------------------------------------------------------------
        np.copyto( dst = sample_obj.x_vec,
                   src = self.vec )
        np.copyto( dst = sample_obj.x_ts,
                   src = self.ts[ i_start : i_end, :] )
        # -----------------------------------------------------------------------------
        i_start += 1
        i_end += 1
        # -----------------------------------------------------------------------------
        np.copyto( dst = sample_obj.y_sold,
                   src = self.ts_sold[ i_start : i_end, :] )
        np.copyto( dst = sample_obj.y_is_on_sale,
                   src = self.ts_is_on_sale[ i_start : i_end, :] )
        np.copyto( dst = sample_obj.z_replenished,
                   src = self.ts_replenished_cat[ i_start : i_end, :] )
        np.copyto( dst = sample_obj.z_returned,
                   src = self.ts_returned_cat[ i_start : i_end, :] )
        np.copyto( dst = sample_obj.z_trashed,
                   src = self.ts_trashed_cat[ i_start : i_end, :] )
        np.copyto( dst = sample_obj.z_found,
                   src = self.ts_found_cat[ i_start : i_end, :] )
        np.copyto( dst = sample_obj.z_missing,
                   src = self.ts_missing_cat[ i_start : i_end, :] )
        # -----------------------------------------------------------------------------
        return

# =====================================================================================
class Sample :
    # ---------------------------------------------------------------------------------
    x_vec = np.array([])
    x_ts = np.array([])
    # ---------------------------------------------------------------------------------
    y_sold = np.array([])
    y_is_on_sale = np.array([])
    # ---------------------------------------------------------------------------------
    z_replenished = np.array([])
    z_returned = np.array([])
    z_trashed = np.array([])
    z_found = np.array([])
    z_missing = np.array([])

# =====================================================================================
class Batch_Specs :
    # ---------------------------------------------------------------------------------
    nb_samples = 0
    timesteps = 0
    vec_dim = 0
    ts_dim = 0

# =====================================================================================
class Batch_Sample :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    X_vec = np.array([])
    X_ts = np.array([])
    # ---------------------------------------------------------------------------------
    Y_sold = np.array([])
    Y_is_on_sale = np.array([])
    # ---------------------------------------------------------------------------------
    Z_replenished = np.array([])
    Z_returned = np.array([])
    Z_trashed = np.array([])
    Z_found = np.array([])
    Z_missing = np.array([])
    # ---------------------------------------------------------------------------------
    sample_obj = Sample()

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, batch_specs = None) :
        # -----------------------------------------------------------------------------
        if batch_specs is None:
            return
        # -----------------------------------------------------------------------------
        self.X_vec = np.zeros( ( batch_specs.nb_samples,
                                 batch_specs.vec_dim),
                                 dtype = 'float32')
        self.X_ts = np.zeros( ( batch_specs.nb_samples,
                                batch_specs.timesteps, batch_specs.ts_dim),
                                dtype = 'float32')
        # -----------------------------------------------------------------------------
        empty_output_tensor = np.zeros( ( batch_specs.nb_samples,
                                          batch_specs.timesteps, 1) )
        # -----------------------------------------------------------------------------
        empty_output_tensor = empty_output_tensor.astype('float32')
        self.Y_sold = empty_output_tensor.copy()
        self.Y_is_on_sale = empty_output_tensor.copy()
        # -----------------------------------------------------------------------------
        empty_output_tensor = empty_output_tensor.astype('uint32')
        self.Z_replenished = empty_output_tensor.copy()
        self.Z_returned = empty_output_tensor.copy()
        self.Z_trashed = empty_output_tensor.copy()
        self.Z_found = empty_output_tensor.copy()
        self.Z_missing = empty_output_tensor.copy()
        # -----------------------------------------------------------------------------
        self.sample_obj = Sample()
        self.sample_obj.x_vec = self.X_vec[0]
        self.sample_obj.x_ts = self.X_ts[0]
        self.sample_obj.y_sold = self.Y_sold[0]
        self.sample_obj.y_is_on_sale = self.Y_is_on_sale[0]
        self.sample_obj.z_replenished = self.Z_replenished[0]
        self.sample_obj.z_returned = self.Z_returned[0]
        self.sample_obj.z_trashed = self.Z_trashed[0]
        self.sample_obj.z_found = self.Z_found[0]
        self.sample_obj.z_missing = self.Z_missing[0]
        # -----------------------------------------------------------------------------
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def copy_sample_obj_into_pos( self, index) :
        # -----------------------------------------------------------------------------
        self.X_vec[ index, :] = self.sample_obj.x_vec
        self.X_ts[ index, :, :] = self.sample_obj.x_ts
        # -----------------------------------------------------------------------------
        self.Y_sold[ index, :, :] = self.sample_obj.y_sold
        self.Y_is_on_sale[ index, :, :] = self.sample_obj.y_is_on_sale
        # -----------------------------------------------------------------------------
        self.Z_replenished[ index, :, :] = self.sample_obj.z_replenished
        self.Z_returned[ index, :, :] = self.sample_obj.z_returned
        self.Z_trashed[ index, :, :] = self.sample_obj.z_trashed
        self.Z_found[ index, :, :] = self.sample_obj.z_found
        self.Z_missing[ index, :, :] = self.sample_obj.z_missing
        # -----------------------------------------------------------------------------
        return

# =====================================================================================
class Timeseries_Categorizer :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    bins = []
    categories = []
    num_categories = 0

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, max_val = 12) :

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.bins = [ -1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12 ]
        increment_steps = 6
        increment_value = 12

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        while self.bins[-1] < max_val :
            # -------------------------------------------------------------------------
            for _ in range( increment_steps) :
                self.bins.append( self.bins[-1] + increment_value )
                if self.bins[-1] >= max_val :
                    break
            # -------------------------------------------------------------------------
            increment_value *= 2

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        self.categories = self.bins[1:]
        self.num_categories = len( self.categories )

        # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_copy( self) :
        return cp.deepcopy(self)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def categorize( self, input_ts) :
        categorical_ts = pd.cut( pd.Series( input_ts[:,0] ), bins = self.bins)
        integer_vector = categorical_ts.cat.codes
        integer_vector = integer_vector.as_matrix().astype('uint32')
        return integer_vector[ :, None]

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_category_at_index( self, index) :
        return self.categories[index]
