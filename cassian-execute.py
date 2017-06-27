#!/usr/bin/python3
import sys, getopt
from cassian.connectivity import DatabaseClient
from cassian.data_management import Dataset

def show_usage() :
    print( 'Cassian-Execute: Usage:' )
    print( './cassian-execute.py -h|--help' )
    print( './cassian-execute.py -l|--list' )
    print( './cassian-execute.py -f|--fetch <store_id> [-r|--resume]' )

def main( argv) :

    long_opts = [ 'help', 'list', 'fetch=', 'resume']

    try :
        opts, args = getopt.getopt( argv, 'hlf:r', long_opts)
    except getopt.GetoptError :
        print( 'Failed to retrieve arguments and/or options!' )
        show_usage()
        sys.exit()

    mode_fetch = False
    resume     = False

    for opt, arg in opts :

        if opt in ( '-h', '--help') :
            show_usage()
            return

        if opt in ( '-l', '--list') :
            client = DatabaseClient()
            client.get_list_of_stores()
            return

        if opt in ( '-f', '--fetch') :
            mode_fetch = True
            store_id = int(arg)

        if opt in ( '-r', '--resume') :
            resume = True

    if mode_fetch :
        client = DatabaseClient( store_id = store_id)
        client.fetch_data( intro_year_limit = 2015,
                           min_num_of_records = 180,
                           reuse_downloaded_result_sets = resume)
        dataset = Dataset( raw_data_file = client.output_file)
        dataset.save()

    return

if __name__ == "__main__":
    main( sys.argv[1:] )
