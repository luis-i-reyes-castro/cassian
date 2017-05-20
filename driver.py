import numpy as np
#from keras.layers import Input
#from keras.models import Model
from cassian.db_clients import DatabaseClient
from cassian.data_management import Dataset
#from cassian.layers import VectorDependentGatedRNN
from cassian.models import CassianModel

# -------------------------------------------------------------------------------------
client = DatabaseClient( store_id = 101)
client.get_store_roster()
#df2 = client.get_table( 'DW_ESTADISTICO_DIM', 100)
#client.download_data( intro_year_limit = 2015, min_num_of_records = 180)

# -------------------------------------------------------------------------------------
#ds = Dataset( store_id = 308)
#ds.save()

#ds = Dataset.load( store_id = 308)
#ds.setup_sampler( nb_samples = 16, timesteps = 90)
#ds.sample_batch()
#
#ds_dic = ds.__dict__

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
#cass = CassianModel( vector_input_dim = 172,
#                     timeseries_input_dim = 15,
#                     replenished_dim = 40)
#cass.plot_model()
#cass.model.summary()
