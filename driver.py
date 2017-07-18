# -------------------------------------------------------------------------------------
from cassian.models import CassianModel

dataset = 'dataset-103/ready-dataset.pkl'
cassian = CassianModel( dataset, batch_size = 16, timesteps = 90)

cassian.model.summary()
cassian.plot_model()

cassian.train_on_dataset()

## -------------------------------------------------------------------------------------
#from cassian.convenience import de_serialize
#from matplotlib import style
#
#results     = de_serialize('results/store-103_results.pkl')
#summary     = results['summary']
#predictions = results['predictions']
#
#def plot_predictions( sku) :
#    style.use('ggplot')
#    return predictions[sku].plot()
