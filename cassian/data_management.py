"""
@author: Luis I. Reyes-Castro

COPYRIGHT

All contributions by Luis I. Reyes-Castro:
Copyright (c) 2017, Luis Ignacio Reyes Castro.
All rights reserved.
"""

# =====================================================================================
import numpy as np
import pandas as pd
from datetime import datetime as dtdt
from sklearn.model_selection import train_test_split
from .convenience import exists_file, ensure_directory
from .convenience import serialize, de_serialize
from .convenience import get_date_today, move_date

# =====================================================================================
class Dataset :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    OUTPUT_DIR  = 'dataset-[STORE-ID]/'
    OUTPUT_FILE = 'ready-dataset.pkl'

    raw_data_file      = None # Must KEEP when splitting
    min_num_of_records = 0 # Must KEEP when splitting

    store_id         = 0 # Must KEEP when splitting
    info_main        = pd.DataFrame() # Must update when splitting
    info_replenish   = pd.DataFrame() # Must update when splitting
    info_other       = pd.DataFrame() # Must update when splitting
    info_description = pd.DataFrame() # Must update when splitting

    num_skus     = 0  # Must update when splitting
    list_of_skus = [] # Must update when splitting
    vectors      = pd.DataFrame() # Must update when splitting
    categorizer  = {} # Must KEEP when splitting
    data         = {} # Must update when splitting

    num_timesteps     = 0 # Must update when splitting
    list_of_sku_probs = [] # Must update when splitting

    vec_dim           = 0 # Must KEEP when splitting
    ts_dim            = 0 # Must KEEP when splitting
    z_replenished_dim = 0 # Must KEEP when splitting
    z_returned_dim    = 0 # Must KEEP when splitting
    z_trashed_dim     = 0 # Must KEEP when splitting
    z_found_dim       = 0 # Must KEEP when splitting
    z_missing_dim     = 0 # Must KEEP when splitting

    vec_mean = np.array([]) # Must update when splitting
    vec_std  = np.array([]) # Must update when splitting
    ts_mean  = np.array([]) # Must update when splitting
    ts_std   = np.array([]) # Must update when splitting

    output_directory = None # Must KEEP(?) when splitting
    output_file      = None # Must update when splitting

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    class TimeseriesCategorizer :

        bins           = []
        categories     = []
        num_categories = 0

        def __init__( self, max_val = 12, min_val_is_one = False) :

            self.bins = [ -1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12 ]

            if min_val_is_one :
                self.bins = self.bins[1:]

            increment_steps = 4
            increment_value = 12

            while self.bins[-1] < max_val :

                for step in range(increment_steps) :

                    self.bins.append( self.bins[-1] + increment_value )

                    if self.bins[-1] >= max_val :
                        break

                increment_value *= 2

            self.categories     = self.bins[1:]
            self.num_categories = len( self.categories )

            return

        def categorize( self, input_ts) :

            categorical_ts = pd.cut( pd.Series( input_ts[:,0] ), bins = self.bins)
            integer_vector = categorical_ts.cat.codes
            integer_vector = integer_vector.as_matrix().astype('uint32')
            return integer_vector[ :, None]

        def get_category_at_index( self, index) :

            return self.categories[index]

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, raw_data_file = None, min_num_of_records = 180) :

        if raw_data_file is None:
            return
        elif not exists_file( raw_data_file) :
            raise ValueError( 'Did not find file:', str(raw_data_file))

        self.raw_data_file      = raw_data_file
        self.min_num_of_records = min_num_of_records

        print( 'Current task: Building dataset object' )
        raw_data = de_serialize( self.raw_data_file)

        self.store_id    = raw_data['store-id']
        big_df           = raw_data['timeseries']
        preselected_skus = raw_data['sku-info']['SKU_A'].tolist()

        print( 'Current task: Selecting SKUs with sufficient timeseries data ' +
               'and processing the timeseries' )

        sku_information_df = raw_data['sku-info']
        sku_timeseries     = {}
        skus_to_drop       = []

        for sku in preselected_skus :

            sku_df = big_df[ big_df['SKU_A'] == sku ]
            sku_df = self.process_sku_timeseries( sku_df)

            if len( sku_df ) >= self.min_num_of_records + 1 :
                sku_timeseries[ sku] = sku_df
            else :
                skus_to_drop.append(sku)

        # serialize( sku_timeseries, 'sku_timeseries.pkl')
        # serialize( skus_to_drop, 'skus_to_drop.pkl')

        # sku_timeseries = de_serialize( 'sku_timeseries.pkl' )
        # skus_to_drop = de_serialize( 'skus_to_drop.pkl' )

        print( 'Current task: Processing SKU static information' )
        self.process_sku_information( sku_information_df,
                                      sku_timeseries,
                                      skus_to_drop)

        self.num_skus     = len( self.info_main )
        self.list_of_skus = self.info_main.index.tolist()

        self.vectors = pd.get_dummies( self.info_main)

        self.categorizer = {}

        self.categorizer['Z1'] = \
        self.TimeseriesCategorizer( self.info_other['REPLENISHED_MAX'].max() )

        self.categorizer['Z2'] = \
        self.TimeseriesCategorizer( self.info_other['RETURNED_MAX'].max() )

        self.categorizer['Z3'] = \
        self.TimeseriesCategorizer( self.info_other['TRASHED_MAX'].max() )

        self.categorizer['Z4'] = \
        self.TimeseriesCategorizer( self.info_other['FOUND_MAX'].max() )

        self.categorizer['Z5'] = \
        self.TimeseriesCategorizer( self.info_other['MISSING_MAX'].max() )

        self.data = {}

        for sku in self.list_of_skus :

            print( 'Building SkuData object for SKU:', str(sku))
            self.data[sku] = SkuData( vector = self.vectors.loc[sku],
                                      timeseries = sku_timeseries[ sku] )

            self.data[sku].z_replenished = \
            self.categorizer['Z1'].categorize( self.data[sku].y_replenished )
            self.data[sku].z_replenished_dim = \
            self.categorizer['Z1'].num_categories

            self.data[sku].z_returned = \
            self.categorizer['Z2'].categorize( self.data[sku].y_returned )
            self.data[sku].z_returned_dim = \
            self.categorizer['Z2'].num_categories

            self.data[sku].z_trashed = \
            self.categorizer['Z3'].categorize( self.data[sku].y_trashed )
            self.data[sku].z_trashed_dim = \
            self.categorizer['Z3'].num_categories

            self.data[sku].z_found = \
            self.categorizer['Z4'].categorize( self.data[sku].y_found )
            self.data[sku].z_found_dim = \
            self.categorizer['Z4'].num_categories

            self.data[sku].z_missing = \
            self.categorizer['Z5'].categorize( self.data[sku].y_missing )
            self.data[sku].z_missing_dim = \
            self.categorizer['Z5'].num_categories

        self.update_num_timesteps_and_list_of_sku_probs()

        arbitrary_sku_obj      = self.data[ self.list_of_skus[0] ]
        self.vec_dim           = arbitrary_sku_obj.vec_dim
        self.ts_dim            = arbitrary_sku_obj.ts_dim
        self.z_replenished_dim = arbitrary_sku_obj.z_replenished_dim
        self.z_returned_dim    = arbitrary_sku_obj.z_returned_dim
        self.z_trashed_dim     = arbitrary_sku_obj.z_trashed_dim
        self.z_found_dim       = arbitrary_sku_obj.z_found_dim
        self.z_missing_dim     = arbitrary_sku_obj.z_missing_dim

        self.compute_mean_std()
        self.normalize_timeseries()

        self.output_directory = \
        self.OUTPUT_DIR.replace( '[STORE-ID]', str(self.store_id))
        ensure_directory( self.output_directory)

        self.output_file = self.output_directory + self.OUTPUT_FILE

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def process_sku_timeseries( self, original_df) :

        df  = pd.DataFrame( original_df )
        sku = df['SKU_A'].iloc[0]

        print( 'Processing timeseries for SKU:', str(sku))

        df.drop( [ 'SKU_A', 'SKU_B'], axis = 1, inplace = True)
        df.set_index( keys = 'DATE_INDEX', inplace = True)
        df.sort_index( inplace = True)

        yesterday = move_date( date = get_date_today(), delta_days = -1)
        if not ( df.index[-1] == yesterday ) :
            last_entry = df.iloc[-1]
            df.loc[ yesterday, 'STOCK_INITIAL'] = last_entry['STOCK_FINAL']
            df.loc[ yesterday, 'SOLD']          = 0
            df.loc[ yesterday, 'REPLENISHED']   = 0
            df.loc[ yesterday, 'TRASHED']       = 0
            df.loc[ yesterday, 'ENTRIES']       = 0
            df.loc[ yesterday, 'ADJUSTMENTS']   = 0
            df.loc[ yesterday, 'STOCK_FINAL']   = last_entry['STOCK_FINAL']
            df.loc[ yesterday, 'STOCK_LIMIT']   = last_entry['STOCK_LIMIT']
            df.loc[ yesterday, 'IS_ON_SALE']    = last_entry['IS_ON_SALE']
            df.loc[ yesterday, 'UNIT_PRICE']    = last_entry['UNIT_PRICE']
            df.loc[ yesterday, 'UNIT_UTILITY']  = last_entry['UNIT_UTILITY']
            df.loc[ yesterday, 'UNIT_COST']     = last_entry['UNIT_COST']

        df['ENTRIES'] -= df['TRASHED']

        rows__ = df['SOLD'] < 0
        df.loc[ rows__, 'TRASHED'] += df.loc[ rows__, 'SOLD']
        df.loc[ rows__, 'SOLD'] = 0

        col_loc = df.columns.tolist().index( 'REPLENISHED' )
        df.insert( loc = col_loc + 1, column = 'RETURNED', value = 0)

        rows__ = df['REPLENISHED'] < 0
        df.loc[ rows__, 'RETURNED'] -= df.loc[ rows__, 'REPLENISHED']
        df.loc[ rows__, 'REPLENISHED'] = 0

        rows__ = df['ENTRIES'] > 0
        df.loc[ rows__, 'REPLENISHED'] += df.loc[ rows__, 'ENTRIES']

        rows__ = df['ENTRIES'] < 0
        df.loc[ rows__, 'RETURNED'] -= df.loc[ rows__, 'ENTRIES']

        df.drop( 'ENTRIES', axis = 1, inplace = True)

        rows__ = df['TRASHED'] < 0
        df.loc[ rows__, 'TRASHED'] *= -1

        col_loc = df.columns.tolist().index( 'ADJUSTMENTS' )
        df.insert( loc = col_loc + 1, column = 'FOUND', value = 0)
        df.insert( loc = col_loc + 2, column = 'MISSING', value = 0)

        rows__ = df['ADJUSTMENTS'] > 0
        df.loc[ rows__, 'FOUND'] += df.loc[ rows__, 'ADJUSTMENTS']

        rows__ = df['ADJUSTMENTS'] < 0
        df.loc[ rows__, 'MISSING'] -= df.loc[ rows__, 'ADJUSTMENTS']

        df.drop( 'ADJUSTMENTS', axis = 1, inplace = True)

        lhs = df['STOCK_FINAL']
        rhs = df['STOCK_INITIAL'] \
            - df['SOLD'] \
            + df['REPLENISHED'] \
            - df['RETURNED'] \
            - df['TRASHED'] \
            + df['FOUND'] \
            - df['MISSING']

        difference = lhs - rhs

        if pd.Series.any( difference != 0 ) :

            rows__ = difference == 2 * ( df['FOUND'] + df['MISSING'] )
            tmp1 = df.loc[ rows__, 'FOUND']
            tmp2 = df.loc[ rows__, 'MISSING']
            df.loc[ rows__, 'FOUND'] = tmp2
            df.loc[ rows__, 'MISSING'] = tmp1
            difference.loc[ rows__] = 0

            if pd.Series.any( difference != 0 ) :
                message = '\tWarning: Numerical inconsistencies on '
                diff_dates = difference[ difference != 0 ].index.tolist()
                for date in diff_dates :
                    message += dtdt.strftime( date, '%Y-%m-%d') + ', '
                print( message[:-2] )

        df = df.asfreq('D')

        columns = [ 'STOCK_INITIAL' ]
        df[ columns] = df[ columns].fillna( method = 'bfill')

        columns = [ 'SOLD', 'REPLENISHED', 'RETURNED',
                    'TRASHED', 'MISSING', 'FOUND' ]
        df[ columns] = df[ columns].fillna( value = 0)

        columns = [ 'STOCK_FINAL', 'STOCK_LIMIT', 'IS_ON_SALE',
                    'UNIT_PRICE', 'UNIT_UTILITY', 'UNIT_COST' ]
        df[ columns] = df[ columns].fillna( method = 'ffill')

        columns = [ 'STOCK_INITIAL', 'STOCK_FINAL' ]
        df[ columns].astype( np.dtype('int32'), copy = False)

        columns = [ 'SOLD', 'REPLENISHED', 'RETURNED',
                    'TRASHED', 'MISSING', 'FOUND', 'STOCK_LIMIT', 'IS_ON_SALE' ]
        df[ columns].astype( np.dtype('uint32'), copy = False)

        columns = [ 'UNIT_PRICE', 'UNIT_UTILITY', 'UNIT_COST' ]
        df[ columns].astype( np.dtype('float32'), copy = False)

        columns = [ 'REPLENISHED', 'RETURNED', 'TRASHED', 'MISSING', 'FOUND' ]
        for col_name in columns :
            new_col_name = col_name + '_FLAG'
            df[ new_col_name] = 0
            df[ new_col_name].astype( np.dtype('uint32'), copy = False)
            df.loc[ df[col_name] != 0, new_col_name] = 1

        date_cols = []
        date_cols.append( ( 'DAY_OF_YEAR',  df.index.dayofyear, 365 ) )
        date_cols.append( ( 'DAY_OF_MONTH', df.index.day,       31  ) )
        date_cols.append( ( 'DAY_OF_WEEK',  df.index.dayofweek, 7   ) )

        for ( col, ts, limit) in date_cols :
            df[ col + '_x' ] = np.cos( 2.0 * np.pi * ts / float(limit) )
            df[ col + '_y' ] = np.sin( 2.0 * np.pi * ts / float(limit) )
            df[ col + '_x' ].astype( np.dtype('float32'), copy = False)
            df[ col + '_y' ].astype( np.dtype('float32'), copy = False)

        return df

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def process_sku_information( self, sku_information_df,
                                       sku_timeseries,
                                       skus_to_drop) :

        df = sku_information_df
        df.set_index( 'SKU_A', inplace = True)
        df.drop( labels = skus_to_drop, axis = 0, inplace = True)

        selected_sections = self.select_sections(df)

        # -----------------------------------------------------------------------------
        section_cols = [ 'SECTION_L0', 'SECTION_L1', 'SECTION_L2' ]
        binary_cols  = [ 'IS_SEASONAL', 'IS_FASHION', 'IS_PERISHABLE' ]

        for ( i, col) in enumerate( section_cols) :
            df[col] = df[col].astype( 'category' )
            df[col] = df[col].cat.set_categories( selected_sections[i] )

        for col in binary_cols :
            positive_rows = df[col].isin( [ 1, 'Y', 'YES', 'S', 'SI'] )
            df.loc[  positive_rows, col] = 1
            df.loc[ ~positive_rows, col] = 0
            df[col] = df[col].astype('category')

        # -----------------------------------------------------------------------------
        replenish_cols = [ 'UNITS_PER_REP', 'STOCK_LIMIT',
                           'UNIT_UTILITY', 'UNIT_COST' ]

        rows__ = df['UNITS_PER_REP'] == 0
        df.loc[ rows__, 'UNITS_PER_REP'] = 1

        df['STOCK_LIMIT']  = 0
        df['UNIT_UTILITY'] = 0
        df['UNIT_COST']    = 0
        sku_ts_cols_drop   = [ 'STOCK_LIMIT', 'UNIT_UTILITY', 'UNIT_COST' ]

        for sku in df.index :

            df.loc[ sku, 'STOCK_LIMIT'] = \
            sku_timeseries[ sku][ 'STOCK_LIMIT'].max()
            df.loc[ sku, 'UNIT_UTILITY'] = \
            sku_timeseries[ sku][ 'UNIT_UTILITY'].mean()
            df.loc[ sku, 'UNIT_COST'] = \
            sku_timeseries[ sku][ 'UNIT_COST'].mean()

            sku_timeseries[ sku].drop( sku_ts_cols_drop, axis = 1, inplace = True)

        rows__ = df['STOCK_LIMIT'] < 4 * df['UNITS_PER_REP']
        df.loc[ rows__, 'STOCK_LIMIT'] = 4 * df.loc[ rows__, 'UNITS_PER_REP']

        rows__ = df['UNIT_UTILITY'] < df['UNIT_COST'] + 0.05
        df.loc[ rows__, 'UNIT_UTILITY'] = df['UNIT_COST'] + 0.05

        # -----------------------------------------------------------------------------
        other_cols = [ 'SOLD_AVG', 'SOLD_STD', 'SOLD_MAX',
                       'REPLENISHED_MAX', 'RETURNED_MAX', 'TRASHED_MAX',
                       'FOUND_MAX', 'MISSING_MAX' ]

        for col_name in other_cols :

            mode_str = col_name[-3:]
            function = None

            if mode_str == 'AVG' :
                function = pd.Series.mean
            elif mode_str == 'STD' :
                function = pd.Series.std
            elif mode_str == 'MAX' :
                function = pd.Series.max
            elif mode_str == 'MIN' :
                function = pd.Series.min

            df[ col_name] = 0

            for sku in df.index :
                df.loc[ sku, col_name] = \
                function( sku_timeseries[ sku][ col_name[:-4] ] )

        # -----------------------------------------------------------------------------
        self.categorizer['UNITS_PER_REP'] = \
        self.TimeseriesCategorizer( df['UNITS_PER_REP'].max(), min_val_is_one = True)

        units_per_rep_array = df[ ['UNITS_PER_REP'] ].as_matrix()
        df['UNITS_PER_REP'] = \
        self.categorizer['UNITS_PER_REP'].categorize( units_per_rep_array )
        df['UNITS_PER_REP'] = df['UNITS_PER_REP'].astype( 'category' )

        df['DAILY_NET_UTILITY'] = \
        ( df['UNIT_UTILITY'] - df['UNIT_COST'] ) * df['SOLD_AVG']

        # -----------------------------------------------------------------------------
        self.info_main      = df[ section_cols + binary_cols + ['UNITS_PER_REP'] ]
        self.info_replenish = df[ replenish_cols + ['DAILY_NET_UTILITY'] ]
        self.info_other     = df[ other_cols ]

        # -----------------------------------------------------------------------------
        description_cols = [ column + '_NAME' for column in section_cols ]
        description_cols.append( 'DESCRIPTION' )

        self.info_description = df[ description_cols ].copy()

        replace = { description_cols[i] : section_cols[i] for i in range(3) }
        self.info_description.rename( columns = replace, inplace = True)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def select_sections( self, sku_information_df) :

        df = self.count_skus_per_section( sku_information_df, level = 0)
        sections_l0 = df[ df['NUM_SKU'] >= 10 ]['SECTION_L0'].tolist()

        df = self.count_skus_per_section( sku_information_df, level = 1)
        sections_l1 = df[ df['NUM_SKU'] >= 10 ]['SECTION_L1'].tolist()

        df = self.count_skus_per_section( sku_information_df, level = 2)
        sections_l2 = df[ df['NUM_SKU'] >= 10 ]['SECTION_L2'].tolist()

        return ( sections_l0, sections_l1, sections_l2)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def count_skus_per_section( self, sku_information_df, level = 0) :

        df = sku_information_df
        columns = [ 'SECTION_L0','SECTION_L1', 'SECTION_L2']
        col_for_counting = 'SKU_B'

        cols_groupby = columns[ : level + 1 ]

        if level in ( 0, 1) :

            df_tmp1 = df.groupby( columns[ : level + 2] ).count().reset_index()
            df_output = df_tmp1.groupby( cols_groupby ).count().reset_index()

            df_tmp2 = df.groupby( cols_groupby ).count().reset_index( drop = True)
            df_output[ col_for_counting] = df_tmp2[ col_for_counting]

            cols_groupby += [ columns[ level + 1 ] ]

        else :

            df_output = df.groupby( cols_groupby ).count().reset_index()

        df_output = df_output.rename( columns = { col_for_counting : 'NUM_SKU' } )

        return df_output[ cols_groupby + [ 'NUM_SKU' ] ]

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __call__( self, sku = None) :

        if sku is None :
            return self.info_main.index.tolist()

        return self.data[sku]

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def save( self) :

        print( 'Saving data to file:', self.output_file)
        serialize( self, self.output_file)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def load( dataset_filename) :

        if not exists_file( dataset_filename) :
            raise ValueError( 'Did not find file:', str(dataset_filename))

        return de_serialize( dataset_filename)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def update_num_timesteps_and_list_of_sku_probs( self, prob_func = 'utility') :

        self.num_timesteps = 0
        self.list_of_sku_probs = [ 0.0 for sku in self.list_of_skus ]

        for sku in self.list_of_skus :
            self.num_timesteps += self.data[sku].num_timesteps

        if prob_func == 'timesteps' :

            for ( i, sku) in enumerate( self.list_of_skus ) :
                self.list_of_sku_probs[i] = \
                float( self.data[sku].num_timesteps) / self.num_timesteps

        elif prob_func == 'utility' :

            total_DNU = self.info_replenish['DAILY_NET_UTILITY'].sum()

            for ( i, sku) in enumerate( self.list_of_skus ) :
                self.list_of_sku_probs[i] = \
                float( self.info_replenish.loc[ sku, 'DAILY_NET_UTILITY'] ) / total_DNU

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def compute_mean_std( self) :

        self.vec_mean = self.vectors.mean( axis = 0).as_matrix()
        self.vec_std  = self.vectors.std( axis = 0).as_matrix()

        self.ts_mean = np.zeros( shape = ( self.ts_dim, ), dtype = 'float32')
        self.ts_std  = np.zeros( shape = ( self.ts_dim, ), dtype = 'float32')

        for ( i, sku) in enumerate( self.list_of_skus ) :
            self.ts_mean += self.list_of_sku_probs[i] \
                          * self.data[sku].get_ts_mean()

        for ( i, sku) in enumerate( self.list_of_skus ) :
            self.ts_std += self.list_of_sku_probs[i] \
                         * self.data[sku].get_ts_mqd_about( self.ts_mean)

        self.ts_std = np.sqrt( self.ts_std )

        self.vec_power = 1.0 / self.vec_mean.clip( min = 0.001)
        self.vec_power = self.vec_power.clip( max = self.vec_power.mean() \
                                                  + 1.5 * self.vec_power.std() )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def normalize_timeseries( self, mode = 'positive-unit') :

        for sku in self.list_of_skus :
            self.data[sku].normalize_inputs( mode, self.vec_power,
                                                   self.ts_mean, self.ts_std )

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def split( self, train_fraction = 0.8):

        skus_training, skus_validation = \
        train_test_split( self.list_of_skus, train_size = train_fraction)

        sku_groups = [ skus_training, skus_validation]
        datasets   = []

        for sku_group in sku_groups :

            ds_obj = Dataset()
            ds_obj.raw_data_file      = self.raw_data_file
            ds_obj.min_num_of_records = self.min_num_of_records

            ds_obj.store_id         = self.store_id
            good_rows               = self.info_main.index.isin(sku_group)
            ds_obj.info_main        = self.info_main.loc[ good_rows]
            ds_obj.info_replenish   = self.info_replenish.loc[ good_rows]
            ds_obj.info_other       = self.info_other.loc[ good_rows]
            ds_obj.info_description = self.info_description.loc[ good_rows]

            ds_obj.num_skus         = len(sku_group)
            ds_obj.list_of_skus     = ds_obj.info_main.index.tolist()
            ds_obj.vectors          = self.vectors.loc[ good_rows]
            ds_obj.categorizer      = self.categorizer.copy()

            ds_obj.data = {}
            for sku in ds_obj.list_of_skus :
                ds_obj.data[sku] = self.data[sku]

            ds_obj.update_num_timesteps_and_list_of_sku_probs()

            ds_obj.vec_dim           = self.vec_dim
            ds_obj.ts_dim            = self.ts_dim
            ds_obj.z_replenished_dim = self.z_replenished_dim
            ds_obj.z_returned_dim    = self.z_returned_dim
            ds_obj.z_trashed_dim     = self.z_trashed_dim
            ds_obj.z_found_dim       = self.z_found_dim
            ds_obj.z_missing_dim     = self.z_missing_dim

            datasets.append(ds_obj)

        datasets[0].compute_mean_std()

        datasets[1].vec_mean = datasets[0].vec_mean
        datasets[1].vec_std  = datasets[0].vec_std
        datasets[1].ts_mean  = datasets[0].ts_mean
        datasets[1].ts_std   = datasets[0].ts_std

        datasets[0].vec_power = self.vec_power
        datasets[1].vec_power = self.vec_power

        datasets[0].normalize_timeseries()
        datasets[1].normalize_timeseries()

        return ( datasets[0], datasets[1])

# =====================================================================================
class SkuData :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    vector        = pd.Series()
    vec           = np.array([])
    vec_dim       = 0
    timeseries    = pd.DataFrame()
    num_timesteps = 0

    ts     = np.array([])
    ts_dim = 0

    y_sold        = np.array([])
    y_is_on_sale  = np.array([])
    y_replenished = np.array([])
    y_returned    = np.array([])
    y_trashed     = np.array([])
    y_found       = np.array([])
    y_missing     = np.array([])

    z_replenished     = np.array([])
    z_replenished_dim = 0
    z_returned        = np.array([])
    z_returned_dim    = 0
    z_trashed         = np.array([])
    z_trashed_dim     = 0
    z_found           = np.array([])
    z_found_dim       = 0
    z_missing         = np.array([])
    z_missing_dim     = 0

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, vector, timeseries) :

        self.vector  = vector
        self.vec     = self.vector.as_matrix().astype('float32')
        self.vec_dim = self.vec.shape[0]

        self.timeseries    = timeseries
        self.num_timesteps = len( self.timeseries ) - 1

        ts_cols = self.timeseries.columns.tolist()
        ts_cols.remove( 'UNIT_PRICE' )
        ts_cols.remove( 'REPLENISHED_FLAG' )
        ts_cols.remove( 'RETURNED_FLAG' )
        ts_cols.remove( 'TRASHED_FLAG' )
        ts_cols.remove( 'FOUND_FLAG' )
        ts_cols.remove( 'MISSING_FLAG' )

        self.ts     = self.timeseries[ ts_cols ].as_matrix().astype('float32')
        self.ts_dim = self.ts.shape[1]

        self.y_sold = \
        self.timeseries[ ['SOLD'] ].as_matrix().astype('float32')

        self.y_is_on_sale = \
        self.timeseries[ ['IS_ON_SALE'] ].as_matrix().astype('float32')

        self.y_replenished = self.timeseries[ ['REPLENISHED'] ].as_matrix()
        self.y_returned    = self.timeseries[ ['RETURNED'] ].as_matrix()
        self.y_trashed     = self.timeseries[ ['TRASHED'] ].as_matrix()
        self.y_found       = self.timeseries[ ['FOUND'] ].as_matrix()
        self.y_missing     = self.timeseries[ ['MISSING'] ].as_matrix()

        return

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_ts_mean( self) :

        return np.mean( self.ts[:-1,:], axis = 0)

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_ts_mqd_about( self, vector) :

        diff = self.ts[:-1,:] - vector
        return np.mean( np.square(diff), axis = 0)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def normalize_inputs( self, mode, vec_power, ts_mean, ts_std) :

        self.vec = self.vec * vec_power

        if mode == 'normal' :
            self.ts -= ts_mean
            self.ts /= ts_std

        elif mode == 'positive-unit' :
            self.ts /= ts_std

        else :
            raise ValueError( 'Invalid normalization mode!' )

        return

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_sample( self, timesteps, seed, batch_sample, sample_index) :

        i_start = int( seed * ( self.num_timesteps - timesteps ) )
        i_end   = i_start + timesteps

        np.copyto( src = self.vec,
                   dst = batch_sample.X_vec[ sample_index, :] )

        np.copyto( src = self.ts[ i_start : i_end, :],
                   dst = batch_sample.X_ts[ sample_index, :, :] )

        i_start += 1
        i_end   += 1

        np.copyto( src = self.y_sold[ i_start : i_end, :],
                   dst = batch_sample.Y_sold[ sample_index, :, :] )

        np.copyto( src = self.y_is_on_sale[ i_start : i_end, :],
                   dst = batch_sample.Y_is_on_sale[ sample_index, :, :] )

        np.copyto( src = self.z_replenished[ i_start : i_end, :],
                   dst = batch_sample.Z_replenished[ sample_index, :, :] )

        np.copyto( src = self.z_returned[ i_start : i_end, :],
                   dst = batch_sample.Z_returned[ sample_index, :, :] )

        np.copyto( src = self.z_trashed[ i_start : i_end, :],
                   dst = batch_sample.Z_trashed[ sample_index, :, :] )

        np.copyto( src = self.z_found[ i_start : i_end, :],
                   dst = batch_sample.Z_found[ sample_index, :, :] )

        np.copyto( src = self.z_missing[ i_start : i_end, :],
                   dst = batch_sample.Z_missing[ sample_index, :, :] )

        return

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_most_recent_inputs( self, timesteps) :

        return ( self.vec, self.ts[ -timesteps :, :] )

    # =---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=---=
    def get_most_recent_data( self, columns, timesteps) :

        if isinstance( columns, str) :
            columns = [ columns ]

        return self.timeseries[columns].iloc[ -timesteps : ]
