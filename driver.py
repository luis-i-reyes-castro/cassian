from cassian.data_management import Dataset
from cassian.models import CassianModel
import numpy as np

# -------------------------------------------------------------------------------------
dataset = Dataset.load( store_id = 101)
dataset.setup_sampler( batch_size = 32, timesteps = 90)

cass = CassianModel( dataset)
cass.plot_model()

#dataset_dict = dataset.as_dictionary()
#cass_dict = cass.as_dictionary()
batch = dataset.sample_batch()

cass.train_on_dataset(1)
