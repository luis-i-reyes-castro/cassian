"""
@author: Luis I. Reyes Castro
"""

# =====================================================================================
import numpy as np
import pandas as pd
import pyodbc
from datetime import datetime as dtdt
from .convenience import exists_file, ensure_directory
from .convenience import de_serialize, serialize
from .convenience import get_date_today, move_date

# =====================================================================================
SQL_SCRIPT  = 'cassian/sql_scripts/tia-netezza_phase-[PHASE].sql'
OUTPUT_DIR  = '/home/luis/cassian/dataset-[STORE-ID]/'
OUTPUT_FILE = 'raw.pkl'

# =====================================================================================
class DatabaseClient :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    sku_timeseries       = None
    sku_information      = None
    sku_info_description = None
    sku_info_main        = None
    sku_info_replenish   = None
    sku_info_other       = None

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__( self, data_source_name = 'NZSQL', store_id = 101) :

        self.data_source = data_source_name
        self.store_id    = store_id

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_list_of_stores( self) :

        print( 'Downloading list of stores...' )

        query   = 'SELECT COD_SUCURSAL, NOM_SUCURSAL, FORMATO, TIPO, ' + \
                  'CIUDAD, FECHA_APERTURA FROM DW_SUCURSAL_DIM' + '\n'
        df      = self.execute_query( query)

        replace = { 'COD_SUCURSAL' : 'ID',
                    'NOM_SUCURSAL' : 'NAME',
                    'FORMATO' : 'FORMAT',
                    'TIPO'  : 'TYPE',
                    'CIUDAD' : 'CITY',
                    'FECHA_APERTURA' : 'OPENING_DATE' }

        df.rename( columns = replace, inplace = True)
        df.sort_values( by = 'ID', inplace = True)

        print( 'Saving data to file list-of-stores.csv.' )
        df.to_csv( 'list-of-stores.csv', index = False)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_columns( self, table) :

        df = self.get_table( table, 1)
        return df.columns.tolist()

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_table( self, table, limit = None) :

        query = 'SELECT * FROM ' + table
        if limit is not None :
            query += ' LIMIT ' + str(limit) + ';'

        return self.execute_query( query)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_info_for_SKU( self, argument) :

        query = 'SELECT * FROM DW_ESTADISTICO_DIM WHERE '
        if isinstance( argument, list) :
            query += 'COD_ESTADISTICO IN ' + str( tuple(argument) )
        else :
            query += 'COD_ESTADISTICO = ' + str(argument)
        query += '\n'

        return self.execute_query( query)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def get_timeseries_for_SKU( self, argument) :

        query = 'SELECT * FROM DW_IPRODUCT_FACT WHERE '
        query += 'FC_COD_SUCURSAL = ' + str( self.store_id) + ' AND '

        if isinstance( argument, list) :
            query += 'FC_COD_ESTADISTICO IN ' + str( tuple(argument) )
        else :
            query += 'FC_COD_ESTADISTICO = ' + str(argument)

        query += ' ORDER BY FC_COD_ESTADISTICO, FC_FECHA' + '\n'

        return self.execute_query( query)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def fetch_data( self, intro_year_limit,
                          min_num_of_records,
                          reuse_downloaded_result_sets = False) :

        self.output_dir = OUTPUT_DIR.replace( '[STORE-ID]', str( self.store_id))
        ensure_directory( self.output_dir)

        self.intro_year_limit = intro_year_limit
        self.dic_replacements = {}
        self.dic_replacements[ '[STORE-ID]' ] = str( self.store_id)
        self.dic_replacements[ '[INTRO-YEAR-LIMIT]' ] = str( self.intro_year_limit)

        df = self.execute_download_phase( 1, reuse_downloaded_result_sets)

        def assert_active_sku( group_of_rows) :

            # N - Nuevo producto; ha sido ingresado al sistema pero todavia
            #     no se despacha.
            # M - Nuevo producto; ya esta siendo despachado.
            # A - Producto activo; ya paso por N y por M.
            # S - Producto suspendido temporalmente; usualmente utilizado para
            #     productos de temporada.
            # I - Producto inactivo; dado de baja en la sucursal.
            # T - Producto inactivo en todas las sucursales.
            # B - Estado post-T que indica que al final del mes no habra
            #     mas registros del producto en ninguna de las sucursales.

            good_rows = group_of_rows['STATE_FLAG'] == 'A'

            if pd.Series.any( good_rows ) :
                blacklist = [ 'S', 'I', 'T', 'B']
                rows__ = group_of_rows['STATE_FLAG'].isin( blacklist)
                return not pd.Series.any( rows__ )

            return False

        print( 'Current task: Preselecting SKUs' )

        df = df.groupby( [ 'SKU_A', 'SKU_B'] ).filter( assert_active_sku)
        df = df[ [ 'SKU_A', 'SKU_B'] ].drop_duplicates()

        preselected_skus = df['SKU_A'].tolist()
        self.dic_replacements[ '[PRESELECTED_SKUS]' ] = str( tuple(preselected_skus) )

        df = self.execute_download_phase( 2, reuse_downloaded_result_sets)

        print( 'Current task: Selecting SKUs with sufficient timeseries data ' +
               'and processing the timeseries' )

        self.sku_timeseries = {}
        for sku in preselected_skus :
            sku_df = df[ df['SKU_A'] == sku ]
            if len( sku_df ) >= min_num_of_records + 1 :
                self.sku_timeseries[ sku] = self.process_sku_timeseries( sku_df)

        selected_skus = self.sku_timeseries.keys()
        self.dic_replacements[ '[SELECTED_SKUS]' ] = str( tuple(selected_skus) )

        df = self.execute_download_phase( 3, reuse_downloaded_result_sets)

        print( 'Current task: Processing SKU static information' )
        self.process_sku_information(df)

        data_object = {}
        data_object['timeseries']       = self.sku_timeseries
        data_object['info-description'] = self.sku_info_description
        data_object['info-main']        = self.sku_info_main
        data_object['info-replenish']   = self.sku_info_replenish
        data_object['info-other']       = self.sku_info_other

        self.output_file = self.output_dir + OUTPUT_FILE
        print( 'Saving data to file:', self.output_file)
        serialize( data_object, self.output_file)

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def execute_download_phase( self, phase, reuse_downloaded_result_sets = False) :

        script = SQL_SCRIPT.replace( '[PHASE]', str(phase))
        query = self.read_sql_script( script)

        if phase == 1 :
            print( 'Current task: Exploring available SKUs' )
        if phase == 2 :
            print( 'Current task: Fetching SKU timeseries (i.e. dynamic info)' )
        if phase == 3 :
            print( 'Current task: Fetching SKU static information' )

        df_file_path = self.output_dir + 'raw_phase-' + str(phase) + '.pkl'

        if reuse_downloaded_result_sets :

            if exists_file( df_file_path) :
                print( 'Found file:', df_file_path )
                print( 'Re-using file to avoid download...' )
                return de_serialize( df_file_path)

            else :
                print( 'Did not find file:', df_file_path )

        df = self.execute_query( query, self.dic_replacements)
        serialize( df, df_file_path)

        return df

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def read_sql_script( self, sql_script) :

        handle = open( sql_script)
        query = handle.read()
        handle.close()

        return query

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def execute_query( self, query, replacements = {}, verbose = False) :

        query_to_execute = query
        for key in replacements :
            query_to_execute = query_to_execute.replace( key, replacements[key])

        print( 'Executing the following query (SQL script):' )
        print( query if not verbose else query_to_execute, end = '' )

        conexion = pyodbc.connect( 'DSN=' + self.data_source )
        df       = None

        try :
            df = pd.read_sql( query_to_execute, conexion)
            print( 'Query execution was successful!' )
        except Exception as some_exception :
            print( 'QUERY execution failed! Error message:', some_exception)

        conexion.close()

        return df

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
    def process_sku_information( self, sku_information_df) :

        df = sku_information_df
        df.set_index( 'SKU_A', inplace = True)
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

        df['STOCK_LIMIT'] = 0
        df['UNIT_UTILITY'] = 0
        df['UNIT_COST'] = 0

        for ( index, _) in df.iterrows() :

            df.loc[ index, 'STOCK_LIMIT'] = \
            self.sku_timeseries[ index][ 'STOCK_LIMIT'].max()
            del self.sku_timeseries[ index][ 'STOCK_LIMIT']

            df.loc[ index, 'UNIT_UTILITY'] = \
            self.sku_timeseries[ index][ 'UNIT_UTILITY'].mean()
            del self.sku_timeseries[ index][ 'UNIT_UTILITY']

            df.loc[ index, 'UNIT_COST'] = \
            self.sku_timeseries[ index][ 'UNIT_COST'].mean()
            del self.sku_timeseries[ index][ 'UNIT_COST']

        rows__ = df['STOCK_LIMIT'] < 4 * df['UNITS_PER_REP']
        df.loc[ rows__, 'STOCK_LIMIT'] = 4 * df.loc[ rows__, 'UNITS_PER_REP']

        # -----------------------------------------------------------------------------
        other_cols = [ 'SOLD_AVG', 'SOLD_STD', 'SOLD_MAX',
                       'REPLENISHED_MAX', 'RETURNED_MAX',
                       'TRASHED_MAX', 'FOUND_MAX', 'MISSING_MAX' ]

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
            for ( index, _) in df.iterrows() :
                df.loc[ index, col_name] = \
                function( self.sku_timeseries[ index][ col_name[:-4] ] )

        # -----------------------------------------------------------------------------
        self.sku_info_main      = df[ section_cols + binary_cols ]
        self.sku_info_replenish = df[ replenish_cols ]
        self.sku_info_other     = df[ other_cols ]

        # -----------------------------------------------------------------------------
        description_cols = [ column + '_NAME' for column in section_cols ]
        description_cols.append( 'DESCRIPTION' )

        self.sku_info_description = df[ description_cols ].copy()

        replace = { description_cols[i] : section_cols[i] for i in range(3) }
        self.sku_info_description.rename( columns = replace, inplace = True)

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
