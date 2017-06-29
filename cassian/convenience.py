"""
@author: Luis I. Reyes Castro
"""

import datetime as dt
from datetime import datetime as dtdt
import numpy as np
import os
import pandas as pd
import pickle

# -------------------------------------------------------------------------------------
def exists_file( filename) :

    return os.path.exists( filename)

# -------------------------------------------------------------------------------------
def ensure_directory( directory) :

    directory += '/' if not directory[-1] == '/' else ''
    directory = os.path.dirname( directory + 'dummy-filename.txt' )

    if not os.path.exists( directory) :
        print( 'Did not find directory', directory)
        print( 'Creating directory:', directory)
        os.makedirs( directory)

    return

# -------------------------------------------------------------------------------------
def serialize( object_to_serialize, filename) :

    filename += '' if filename[-4:] == '.pkl' else '.pkl'

    if isinstance( object_to_serialize, pd.DataFrame) :
        object_to_serialize.to_pickle( filename)

    else :
        handle = open( filename, 'wb')
        pickle.dump( object_to_serialize, handle, pickle.HIGHEST_PROTOCOL)
        handle.close()

    return

# -------------------------------------------------------------------------------------
def de_serialize( filename) :

    filename += '' if filename[-4:] == '.pkl' else '.pkl'

    handle = open( filename, 'rb')
    obj = pickle.load( handle)
    handle.close()

    return obj

# -------------------------------------------------------------------------------------
def get_date_today() :

    return dtdt.today().date()

# -------------------------------------------------------------------------------------
def move_date( date, delta_days) :

    return date + dt.timedelta( days = delta_days)

# -------------------------------------------------------------------------------------
def save_df_to_excel( dataframe, output_file, output_sheet = 'Sheet') :

    writer = pd.ExcelWriter(output_file)
    dataframe.to_excel( writer, output_sheet)
    writer.save

    return
