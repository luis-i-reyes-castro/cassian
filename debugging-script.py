# -------------------------------------------------------------------------------------
from cassian.data_management import Dataset
from cassian.models import CassianModel

dataset = 'dataset-103/raw-data.pkl'
ds = Dataset(dataset)
ds.save()

#dataset = 'dataset-103/ready-dataset.pkl'
#cassian = CassianModel( dataset, batch_size = 16, timesteps = 90)
#
#vec_mean = cassian.dataset.vec_mean
#powers = 1 / vec_mean
#threshold = powers.mean() + 1.5 * powers.std()
#powers = powers.clip( max = threshold)
#
#product = cassian.dataset()[0]
#product_obj = cassian.dataset(product)
#vec = product_obj.vec
#
#sample_vec = vec * powers
#
#cassian.model.summary()
#cassian.plot_model()
#
#cassian.train_on_dataset()

## -------------------------------------------------------------------------------------
#from cassian.connectivity import DatabaseClient
#from cassian.data_management import Dataset
#
#store_id = 443
#
#client = DatabaseClient( store_id = store_id)
##client.fetch_data( intro_year_limit = 2016, min_num_of_records = 180)
#
#dataset = Dataset( raw_data_file = client.output_file)
##preselected_skus = dataset.preselected_skus
#dataset.save()
