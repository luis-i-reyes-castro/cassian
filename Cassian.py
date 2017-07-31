#!/usr/bin/python3
import sys, getopt

def show_usage() :
    print( '========================================' )
    print( 'CASSIAN Decision Support System - Usage:' )
    print( '----------------------------------------' )
    print( '(+) Print this manual:' )
    print( '    ./Cassian.py -h|--help' )
    print( '(+) Print a list of stores:' )
    print( '    ./Cassian.py -l|--list' )
    print( '(+) Fetch data for training or prediction:' )
    print( '    ./Cassian.py -s|--store <store_id> -f|--fetch [--build-dataset_only]' )
    print( '(+) Train a new model or load and re-train a saved one:' )
    print( '    ./Cassian.py -s|--store <store_id> -T|--train [-l|--load] ' +
           '[-e|--epochs] <num_of_epochs> ' +
           '[-b|--batch-size] <batch_size> ' +
           '[-t|--timesteps] <timesteps> ' +
           '[-w|--workers] <num_of_workers> ' )
    print( '(+) Compute predictions and generate store summary:' )
    print( '    ./Cassian.py -s|--store <store_id> -P|--predict' )
    print( '(+) Plot history of stock, sales and sale predictions for an SKU:' )
    print( '    ./Cassian.py -s|--store <store_id> -p|--plot <SKU>' )
    print( '====================================================' )

def main( argv) :

    opts_short = 'hls:fTl:e:b:t:w:R:L:Pp:'
    opts_long  = [ 'help', 'list', 'store=',
                   'fetch', 'build-dataset_only', 'train', 'load=',
                   'epochs=', 'batch_size=', 'timesteps=', 'workers=',
                   'regularization=', 'learning-rate=',
                   'predict', 'plot=' ]

    try :
        opts, args = getopt.getopt( argv, opts_short, opts_long)

    except getopt.GetoptError :
        print( 'Failed to retrieve arguments and/or options!' )
        show_usage()
        sys.exit()

    store_id     = 0
    mode_fetch   = False
    build_only   = False
    mode_train   = False
    mode_load    = False
    epochs       = 1
    batch_size   = 32
    timesteps    = 90
    workers      = 4
    regularize   = 1E-3
    learn_rate   = 0.002
    mode_predict = False
    mode_plot    = False
    sku_to_plot  = None

    for opt, arg in opts :

        if opt in ( '-h', '--help') :
            show_usage()
            return

        if opt in ( '-l', '--list') :

            from cassian.connectivity import DatabaseClient

            client = DatabaseClient()
            client.get_list_of_stores()

            return

        if opt in ( '-s', '--store') :
            store_id = int(arg)
        if opt in ( '-f', '--fetch') :
            mode_fetch = True
        if opt in ( '--build-dataset_only') :
            build_only = True
        if opt in ( '-T', '--train') :
            mode_train = True
        if opt in ( '-l', '--load') :
            mode_load = True
            model_to_load = str(arg)
        if opt in ( '-e', '--epochs') :
            epochs = max( ( epochs, int(arg) ) )
        if opt in ( '-b', '--batch-size') :
            batch_size = max( ( batch_size, int(arg) ) )
        if opt in ( '-t', '--timesteps') :
            timesteps = max( ( timesteps, int(arg) ) )
        if opt in ( '-w', '--workers') :
            workers = max( ( 2, int(arg) ) )
        if opt in ( '-R', '--regularize') :
            regularize = max( ( 0.0, float(arg) ) )
        if opt in ( '-L', '--learning-rate') :
            learn_rate = max( ( 1E-5, float(arg) ) )
        if opt in ( '-P', '--predict') :
            mode_predict = True
        if opt in ( '-p', '--plot') :
            mode_plot   = True
            sku_to_plot = int(arg)

    if mode_fetch :

        from cassian.connectivity import DatabaseClient
        from cassian.data_management import Dataset

        client = DatabaseClient( store_id = store_id)

        if not build_only :
            client.fetch_data( intro_year_limit = 2016)

        dataset = Dataset( raw_data_file = client.output_file,
                           min_num_of_records = 180)
        dataset.save()

    if mode_train :

        from cassian.data_management import Dataset
        from cassian.models import CassianModel

        if mode_load :
            cassian = CassianModel.load( model_to_load)
        else :
            dataset_file = Dataset.OUTPUT_DIR + Dataset.OUTPUT_FILE
            dataset      = dataset_file.replace( '[STORE-ID]', str(store_id))
            cassian      = CassianModel( dataset, batch_size, timesteps,
                                         regularization = regularize,
                                         learning_rate = learn_rate )

        cassian.plot_model()
        cassian.train_on_dataset( epochs = epochs, workers = workers)

    if mode_predict :

        from cassian.models import CassianModel

        cassian = CassianModel.load( model_to_load)
        cassian.compute_predictions()

    if mode_plot :

        from cassian.models import CassianModel
        from cassian.convenience import de_serialize
        import matplotlib.pyplot as plt

        results_file = CassianModel.RESULTS_DIR + CassianModel.RESULTS_FILE
        results_file = results_file.replace( '[STORE-ID]', str(store_id))
        results      = de_serialize( results_file)
        predictions  = results['predictions']

        plt.style.use('ggplot')
        predictions[sku_to_plot].plot()
        plt.show()

    return

if __name__ == "__main__":
    main( sys.argv[1:] )
