## -------------------------------------------------------------------------------------
#from cassian.connectivity import DatabaseClient
#
#client = DatabaseClient( store_id = 101)
#client.fetch_data( intro_year_limit = 2015,
#                   min_num_of_records = 180, reuse_downloaded_result_sets=True)

# -------------------------------------------------------------------------------------
from cassian.models import CassianModel

dataset = 'dataset-101/ready-dataset.pkl'
cassian = CassianModel( dataset, batch_size = 16, timesteps = 90)

cassian.train_on_dataset( epochs = 20, workers = 8)
cassian.save()
cassian.train_on_dataset( epochs = 20, workers = 8)
cassian.save()
cassian.train_on_dataset( epochs = 20, workers = 8)
cassian.save()
cassian.compute_predictions()

## -------------------------------------------------------------------------------------
#from cassian.models import CassianModel
#
#model_file = 'trained-models/store-101-model.pkl'
#cassian = CassianModel.load( model_file)
#
#cassian.train_on_dataset( epochs = 60, workers = 8)
#cassian.save()
#cassian.compute_predictions()
