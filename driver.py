## -------------------------------------------------------------------------------------
#from cassian.connectivity import DatabaseClient
#
#client = DatabaseClient( store_id = 101)
#client.fetch_data( intro_year_limit = 2015,
#                   min_num_of_records = 180, reuse_downloaded_result_sets=True)

# -------------------------------------------------------------------------------------
from cassian.data_management import Dataset
from cassian.models import CassianModel
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime as dtdt

dataset = Dataset.load( store_id = 101)
cass = CassianModel( dataset, batch_size = 16, timesteps = 90)

df = dataset.data[2012000002].get_most_recent_data( 'SOLD', 90)
df = pd.DataFrame(df)
vec = dataset.data[2012000002].get_most_recent_inputs(90)[1][:,1]

second = df.index[1]
last   = df.index[-1]

df.loc[ second : last, 'PRED'] = vec[:-1]
df.loc[ last + dt.timedelta( days = 1), 'PRED'] = vec[-1]

df.plot()

#outputs = cass.compute_predictions()

#cass.train_on_dataset(1)
