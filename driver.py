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
#X_vec = Input( batch_shape = ( 10, 32) )
#X_ts  = Input( batch_shape = ( 10, None, 24) )
#
#vdrnn = VectorDependentGatedRNN( units = 4,
#                                 stateful = True,
#                                 learn_initial_state_bias = False,
#                                 learn_initial_state_kernel = False,
#                                 architecture ='single-gate')
#Y_ts = vdrnn( inputs = [ X_vec, X_ts])
#model = Model( inputs = [ X_vec, X_ts], outputs = Y_ts)
#
#x_vec_batch = np.zeros( ( 10, 32) )
#x_ts_batch  = np.zeros( ( 10, 20, 24) )
#
#vdrnn.compute_initial_states( np.random.randn( 10, 32) )
#y_ts_batch  = model.predict( [ x_vec_batch, x_ts_batch])
#y_ts_batch2 = model.predict( [ x_vec_batch, x_ts_batch])
#
#model.summary()

# -------------------------------------------------------------------------------------
#cass = CassianModel( dataset.batch_specs)
#cass.plot_model()
#cass.model.summary()
