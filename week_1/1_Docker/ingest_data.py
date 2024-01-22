
# Import packages
import pandas as pd
import os
from time import time
import argparse
from sqlalchemy import create_engine

def main(params):
    user = params.user
    password = params.password
    host = params.host
    port = params.port
    db = params.db
    table_name = params.table_name
    url = params.url
    gz_name = "output.csv.gz"
    csv_name = "output.csv"

    # Download data from url and write to csv using command line
    os.system(f"wget {url} -O {gz_name}")
    os.system(f"gzip -d {csv_name}")

    engine = create_engine('postgresql://root:root@pgdatabase:5432/ny_taxi')

    #over 1M rows in ny taxi data, upload data to db in batches
    df_iter = pd.read_csv(csv_name, iterator=True, chunksize=100000)

    # now insert all data
    for df in df_iter:
        start_ts = time()
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

        df.to_sql(name=table_name, con=engine, if_exists='append')
        end_ts = time()

        print('inserted another chunk in %.3f seconds' % (end_ts - start_ts))

# parse python code to command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ingest CSV data to Postgres')
    # arguments: user, password, host, port, db name, table name, url of csv
    parser.add_argument('--user', help='user name for postgres')
    parser.add_argument('--password', help='password for postgres')
    parser.add_argument('--host', help='host for postgres')
    parser.add_argument('--port', help='port for postgres')
    parser.add_argument('--db', help='database for postgres')
    parser.add_argument('--table_name', help='table name where we will write results')
    parser.add_argument('--url', help='url of the csv file')
    # create db
    args = parser.parse_args()
    main(args)

