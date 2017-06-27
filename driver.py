## -------------------------------------------------------------------------------------
#from cassian.connectivity import DatabaseClient
#
#client = DatabaseClient( store_id = 101)
#client.fetch_data( intro_year_limit = 2015,
#                   min_num_of_records = 180, reuse_downloaded_result_sets=True)

# -------------------------------------------------------------------------------------
from cassian.models import CassianModel

dataset = 'dataset-101/ready-dataset.pkl'
cassian = CassianModel( dataset, batch_size = 24, timesteps = 90)

cassian.train_on_dataset( epochs = 2, workers = 8)
cassian.compute_predictions()
cassian.save()
