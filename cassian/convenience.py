"""
@author: Luis I. Reyes Castro
"""

import matplotlib.pyplot as plt
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
def plot_timeseries( ts_1, ts_2) :

    COLOR_TS_1 = 'blue'
    COLOR_TS_2 = 'red'
    LINEWIDTH  = 0.8
    MARKERSIZE = 4

    time = np.arange( len(ts_1) ) + 1

    plt.plot( time, ts_1, label = 'ts_{1}', \
              linestyle = '-', linewidth = LINEWIDTH, color = COLOR_TS_1, \
              marker = 'o', markerfacecolor = COLOR_TS_1, \
              markeredgecolor = 'None', markersize = MARKERSIZE)

    plt.plot( time, ts_2, label = 'ts_{2}', \
              linestyle = '-', linewidth = LINEWIDTH, color = COLOR_TS_2, \
              marker = 'o', markerfacecolor = COLOR_TS_2, \
              markeredgecolor = 'None', markersize = MARKERSIZE)

    plt.show()

#    plt.grid()
#    plt.xlim( ( -1, len(ts_x1)) )
#    plt.ylim( ( -1.10, +1.10) )
#    plt.xlabel( 't', fontsize = FONTSIZE_AXIS_LABEL )
#    plt.ylabel( 'x_1(t), x_2(t), y(t)', fontsize = FONTSIZE_AXIS_LABEL )
#    plt.legend( fontsize = FONTSIZE_LEGEND )

#    if output_file is not None :
#        plt.savefig( output_file, format = 'eps', dpi = 300, bbox_inches ='tight')
#    plt.show()
