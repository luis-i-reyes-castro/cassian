import sys, getopt
from cassian.db_clients import DatabaseClient

def show_usage() :
    print( 'Stage-01 Script - Usage:' )
    print( 'python3 run-stage-01.py -h|--help' )
    print( 'python3 run-stage-01.py -l|--list' )
    print( 'python3 run-stage-01.py -s|--store <store_id>' )

def main(argv):
    try :
        opts, args = getopt.getopt( argv, 'hls:', [ 'help', 'list', 'store='] )
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
        elif opt in ( '-s', '--store') :
            client = DatabaseClient( store_id = int(arg) )
            client.download_data( intro_year_limit = 2015, min_num_of_records = 180)

if __name__ == "__main__":
    main( sys.argv[1:] )
