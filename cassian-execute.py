#!/usr/bin/python3
import sys, getopt
from cassian.connectivity import DatabaseClient
from cassian.data_management import Dataset

def show_usage() :
    print( 'Stage-01 Script - Usage:' )
    print( 'python3 run-stage-01.py -h|--help' )
    print( 'python3 run-stage-01.py -l|--list' )
    print( 'python3 run-stage-01.py -f|--fetch <store_id>' )
    print( 'python3 run-stage-01.py -b|--build <store_id>' )

def main( argv) :
    try :
        opts, args = getopt.getopt( argv, 'hlf:b:',
                                    [ 'help', 'list', 'fetch=', 'build='] )
    except getopt.GetoptError :
        print( 'Failed to retrieve arguments and/or options!' )
        show_usage()
        sys.exit()
    for opt, arg in opts :
        if opt in ( '-h', '--help') :
            show_usage()
        elif opt in ( '-l', '--list') :
            client = DatabaseClient()
            client.get_list_of_stores()
        elif opt in ( '-f', '--fetch') :
            client = DatabaseClient( store_id = int(arg) )
            client.fetch_data( intro_year_limit = 2015, min_num_of_records = 180)
        elif opt in ( '-b', '--build') :
            dataset = Dataset( store_id = int(arg) )
            dataset.save()

if __name__ == "__main__":
    main( sys.argv[1:] )
