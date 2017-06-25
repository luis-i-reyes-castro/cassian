## -------------------------------------------------------------------------------------
#from cassian.connectivity import DatabaseClient
#
#client = DatabaseClient( store_id = 101)
#client.fetch_data( intro_year_limit = 2015,
#                   min_num_of_records = 180, reuse_downloaded_result_sets=True)

# -------------------------------------------------------------------------------------
from cassian.data_management import Dataset
from cassian.models import CassianModel
from cassian.convenience import move_date
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime as dtdt

batch_size = 16
timesteps = 90
sku = 2012000002

dataset = Dataset.load( store_id = 101)
cass = CassianModel( dataset, batch_size = batch_size, timesteps = timesteps)

#df = dataset.data[sku].get_most_recent_data( 'SOLD', timesteps)
#vec = dataset.data[sku].get_most_recent_inputs(90)[1][:,1]
#
#start = move_date( date = df.index[0], delta_days = +1)
#end   = df.index[-1]
#df.loc[ start : end, 'PRED'] = vec[:-1]
#
#end = move_date( date = df.index[-1], delta_days = +1)
#df.loc[ end, 'PRED'] = vec[-1]
#
#df.plot()

cass.train_on_dataset(1)
predictions_dict = cass.compute_predictions()

