from cassian.data_management import Dataset
from cassian.models import CassianModel, batch_generator
import numpy as np

# -------------------------------------------------------------------------------------
dataset = Dataset.load( store_id = 101)
cass = CassianModel( dataset, batch_size = 16, timesteps = 90)

cass.train_on_dataset(10)

#dataset_dict = dataset.as_dictionary()
#cass_dict = cass.as_dictionary()
#batch = dataset.sample_batch()

