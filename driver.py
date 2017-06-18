from cassian.data_management import Dataset
from keras.layers import Input
from keras.models import Model
from cassian.layers import VectorDependentGatedRNN
from cassian.models import CassianModel
import numpy as np

# -------------------------------------------------------------------------------------
#dataset = Dataset( store_id = 101)
#dataset.save()

dataset = Dataset.load( store_id = 101)
dataset.setup_sampler( batch_size = 32, timesteps = 90)

dataset_dict = dataset.as_dictionary()
batch = dataset.sample_batch()



# -------------------------------------------------------------------------------------
cass = CassianModel( dataset.batch_specs)
cass.plot_model()
cass.model.summary()
