# Source libraries
import os
import logging
from datetime import datetime
import pandas as pd
# pip install apache-airflow
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# pip install google-cloud-storage
from google.cloud import storage
# pip install apache-airflow-providers-google
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateExternalTableOperator
#import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

# Get GCP input data
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
BUCKET = os.environ.get("GCP_GCS_BUCKET")

# Get file structure data
local_data_path = "/.project/data/raw/Mendeley_data/" # this will need to change when dockerized
temp_path = "/.project/data/raw/temp/"
local_data_file = "100_Batches_IndPenSim_V3.csv"
path_to_local_home = os.environ.get("AIRFLOW_HOME", "/opt/airflow/")
gcs_path = "raw/"

# Send raw data as series of parquet files to GCS
# NOTE: Can take several minutes depending on internet speed
def raw_to_gcs(gcs_bucket, gcs_path, local_data_file, local_data_path, temp_path):
    chunk_size = 10000
    next_id = 1

    # Loop through csv and send chunks of data as parquet files to gcs
    for chunk in pd.read_csv(local_data_path+local_data_file, sep=",", chunksize=chunk_size):
        # add unique id to data records
        nrow = chunk.shape[0]
        ids = list(range(next_id,next_id+nrow))
        chunk["id"] = ids

        # convert df to parquet file
        parquet_file = f"{temp_path}insulin_batch_set_{ids[0]}.parquet"
        chunk.to_parquet(parquet_file, engine = 'pyarrow')

        # upload to gcs
        client = storage.Client()
        bucket = client.get_bucket(gcs_bucket)
        bucket_path = os.path.join(gcs_path, os.path.basename(parquet_file))
        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(parquet_file)

        # set next id
        next_id = ids[len(ids)-1] + 1


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 12),
    "depends_on_past": False,
    "retries": 1
}

# NOTE: DAG declaration - using a Context Manager (an implicit way)
with DAG(
    dag_id="raw_data_ingestion_gcs_dag",
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
) as dag:

    task = PythonOperator(
        task_id='raw_to_gcs_task',
        python_callable=raw_to_gcs,
        op_args=[
            BUCKET,
            gcs_path,
            local_data_file,
            local_data_path,
            temp_path
        ]
    )

    task
    