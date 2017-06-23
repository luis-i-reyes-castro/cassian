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

dataset = Dataset.load( store_id = 101)
cass = CassianModel( dataset, batch_size = 16, timesteps = 90)

#cass.train_on_dataset(1)
