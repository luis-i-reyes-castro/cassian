"""
@author: Luis I. Reyes Castro
"""

# =====================================================================================
import pandas as pd
import pyodbc
from .convenience import exists_file, ensure_directory
from .convenience import de_serialize, serialize
from .convenience import save_df_to_excel

# =====================================================================================
class DatabaseClient :

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    SQL_SCRIPT  = '/home/luis/cassian/cassian/sql_scripts/tia-netezza_phase-[PHASE].sql'
    OUTPUT_DIR  = '/home/luis/cassian/dataset-[STORE-ID]/'
    OUTPUT_FILE = 'raw-data.pkl'

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

        self.output_dir = self.OUTPUT_DIR.replace( '[STORE-ID]', str( self.store_id))
        ensure_directory( self.output_dir)

        self.output_file = self.output_dir + self.OUTPUT_FILE

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
    def get_list_of_stores( self) :

        print( 'Downloading list of stores...' )

        query   = 'SELECT COD_SUCURSAL, NOM_SUCURSAL, FORMATO, TIPO, ' + \
                  'CIUDAD, FECHA_APERTURA FROM DW_SUCURSAL_DIM' + '\n'
        df      = self.execute_query( query)

        replace = { 'COD_SUCURSAL'   : 'STORE-ID',
                    'NOM_SUCURSAL'   : 'NAME',
                    'FORMATO'        : 'FORMAT',
                    'TIPO'           : 'TYPE',
                    'CIUDAD'         : 'CITY',
                    'FECHA_APERTURA' : 'OPENING_DATE' }

        df.rename( columns = replace, inplace = True)

        df.set_index( keys = 'STORE-ID', inplace = True)
        df.sort_index( inplace = True)

        print( 'Saving data to file List-of-Stores.xlsx.' )
        save_df_to_excel( df, 'List-of-Stores.xlsx', 'Stores')

        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def fetch_data( self, intro_year_limit,
                          reuse_downloaded_result_sets = False) :

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

        df_timeseries = self.execute_download_phase( 2, reuse_downloaded_result_sets)
        df_sku_info   = self.execute_download_phase( 3, reuse_downloaded_result_sets)

        data_object = {}
        data_object['store-id']   = self.store_id
        data_object['sku-info']   = df_sku_info
        data_object['timeseries'] = df_timeseries

        print( 'Saving data to file:', self.output_file)
        serialize( data_object, self.output_file)

        return data_object

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def execute_download_phase( self, phase,
                                reuse_downloaded_result_sets = False,
                                serialize_for_debugging = False ) :

        script = self.SQL_SCRIPT.replace( '[PHASE]', str(phase))
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

        if serialize_for_debugging :
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
