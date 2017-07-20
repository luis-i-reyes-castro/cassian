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
    print( '    ./Cassian.py -s|--store <store_id> -f|--fetch [-r|--resume_fetch]' )
    print( '(+) Train a new model or load and re-train a saved one:' )
    print( '    ./Cassian.py -s|--store <store_id> -T|--train [-l|--load] ' +
           '[-e|--epochs] <num_of_epochs> ' +
           '[-b|--batch_size] <batch_size> ' +
           '[-t|--timesteps] <timesteps> ' +
           '[-w|--workers] <num_of_workers> ' )
    print( '(+) Compute predictions and generate store summary:' )
    print( '    ./Cassian.py -s|--store <store_id> -P|--predict' )
    print( '(+) Plot history of stock, sales and sale predictions for an SKU:' )
    print( '    ./Cassian.py -s|--store <store_id> -p|--plot <SKU>' )
    print( '====================================================' )

def main( argv) :

    opts_short = 'hls:frlTe:b:t:w:Pp:'
    opts_long  = [ 'help', 'list', 'store=',
                   'fetch', 'resume_fetch', 'load', 'train',
                   'epochs=', 'batch_size=', 'timesteps=', 'workers=',
                   'predict', 'plot=' ]

    model_file   = 'trained-models/store-[STORE-ID]_model.pkl'
    dataset_file = 'dataset-[STORE-ID]/ready-dataset.pkl'
    results_file = 'results/store-[STORE-ID]_results.pkl'

    try :
        opts, args = getopt.getopt( argv, opts_short, opts_long)

    except getopt.GetoptError :
        print( 'Failed to retrieve arguments and/or options!' )
        show_usage()
        sys.exit()

    store_id     = 0
    mode_fetch   = False
    resume       = False
    mode_load    = False
    mode_train   = False
    epochs       = 1
    batch_size   = 32
    timesteps    = 90
    workers      = 2
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

        if opt in ( '-r', '--resume_fetch') :
            resume = True

        if opt in ( '-l', '--load') :
            mode_load = True

        if opt in ( '-T', '--train') :
            mode_train = True

        if opt in ( '-e', '--epochs') :
            epochs = max( ( epochs, int(arg) ) )

        if opt in ( '-b', '--batch_size') :
            batch_size = max( ( batch_size, int(arg) ) )

        if opt in ( '-t', '--timesteps') :
            timesteps = max( ( timesteps, int(arg) ) )

        if opt in ( '-w', '--workers') :
            workers = max( ( 2, int(arg) ) )

        if opt in ( '-P', '--predict') :
            mode_predict = True

        if opt in ( '-p', '--plot') :
            mode_plot   = True
            sku_to_plot = int(arg)

    if not store_id and \
    ( mode_fetch or mode_load or mode_train or mode_predict or mode_plot ) :

        print( 'Error: Cannot fetch, load, train or predict without a Store ID!' )
        show_usage()
        sys.exit()

    if mode_fetch :

        from cassian.connectivity import DatabaseClient
        from cassian.data_management import Dataset

        client = DatabaseClient( store_id = store_id)
        client.fetch_data( intro_year_limit = 2016,
                           reuse_downloaded_result_sets = resume)

        dataset = Dataset( raw_data_file = client.output_file,
                           min_num_of_records = 180)
        dataset.save()

    if mode_train :

        from cassian.models import CassianModel

        if mode_load :
            model_file = model_file.replace( '[STORE-ID]', str(store_id))
            cassian    = CassianModel.load( model_file)
        else :
            dataset = dataset_file.replace( '[STORE-ID]', str(store_id))
            cassian = CassianModel( dataset, batch_size, timesteps)

        cassian.train_on_dataset( epochs = epochs, workers = workers)

    if mode_predict :

        from cassian.models import CassianModel

        model_file = model_file.replace( '[STORE-ID]', str(store_id))
        cassian    = CassianModel.load( model_file)

        cassian.compute_predictions()

    if mode_plot :

        from cassian.convenience import de_serialize
        import matplotlib.pyplot as plt

        results_file = results_file.replace( '[STORE-ID]', str(store_id))
        results      = de_serialize( results_file)
        predictions  = results['predictions']

        plt.style.use('ggplot')
        predictions[sku_to_plot].plot()
        plt.show()

    return

if __name__ == "__main__":
    main( sys.argv[1:] )
