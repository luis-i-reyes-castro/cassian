#!/usr/bin/python3
import sys, getopt

def show_usage() :
    print( 'CASSIAN - Usage:' )
    print( './Cassian.py -h|--help' )
    print( './Cassian.py -l|--list' )
    print( './Cassian.py -s|--store <store_id> ' +
           '[-f|--fetch] [-r|--resume_fetch] [-l|--load] [-T|--train] ' +
           '[-e|--epochs] <num_of_epochs> ' +
           '[-b|--batch_size] <batch_size> ' +
           '[-t|--timesteps] <timesteps> ' +
           '[-w|--workers] <num_of_workers> ' +
           '[-p|--predict]' )

def main( argv) :

    opts_short = 'hls:frlTe:b:t:w:p'
    opts_long  = [ 'help', 'list', 'store=',
                   'fetch', 'resume_fetch', 'load', 'train',
                   'epochs=', 'batch_size=', 'timesteps=', 'workers=', 'predict' ]

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
    batch_size   = 16
    timesteps    = 90
    workers      = 2
    mode_predict = False

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

        if opt in ( '-p', '--predict') :
            mode_predict = True

    if not store_id and ( mode_fetch or mode_load or mode_train or mode_predict ) :
        print( 'Error: Cannot fetch, load, train or predict without a Store ID!' )
        show_usage()
        sys.exit()

    if mode_fetch :

        from cassian.connectivity import DatabaseClient
        from cassian.data_management import Dataset

        client = DatabaseClient( store_id = store_id)
        client.fetch_data( intro_year_limit = 2015,
                           min_num_of_records = 180,
                           reuse_downloaded_result_sets = resume)

        dataset = Dataset( raw_data_file = client.output_file)
        dataset.save()

    if mode_load or mode_train or mode_predict :

        from cassian.models import CassianModel

        if mode_load :
            model_file = 'trained-models/store-' + str(store_id) + '_model.pkl'
            cassian = CassianModel.load( model_file)
        else :
            dataset = 'dataset-' + str(store_id) + '/ready-dataset.pkl'
            cassian = CassianModel( dataset, batch_size, timesteps)

        if mode_train :
            cassian.train_on_dataset( epochs = epochs, workers = workers)
        else :
            cassian.compute_predictions()

    return

if __name__ == "__main__":
    main( sys.argv[1:] )
