#--- Function File for all functions used by raw_data_to_gcs dag ---
import os
import sys
import logging
import subprocess
from datetime import datetime, timedelta
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
# pip install pyspark
import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Send Processed Data to BigQuery
def processed_to_bq(gcs_bucket, gcs_raw_path, gcs_sample_path, project_id, credentials_location, spark_jar_path, full_backfill=False):
    # start spark standalone instance with worker
    start_spark_master = "cd $SPARK_HOME && ./sbin/start-master.sh --port 7078"
    start_spark_worker = "cd $SPARK_HOME && ./sbin/start-worker.sh spark://127.0.0.1:7078"
    start_master_process = subprocess.Popen(start_spark_master, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    start_master_output, start_master_error = start_master_process.communicate()
    start_worker_process = subprocess.Popen(start_spark_worker, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    start_worker_output, start_worker_error = start_worker_process.communicate()
    logging.info("spark master + worker started")

    # define spark configuration
    conf = SparkConf() \
        .setMaster("spark://127.0.0.1:7078") \
        .setAppName("process_raw_data") \
        .set("spark.jars", spark_jar_path) \
        .set("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .set("spark.hadoop.google.cloud.auth.service.account.json.keyfile", credentials_location)
    logging.info("spark config created")

    # set up spark context
    sc = SparkContext(conf=conf)
    hadoop_conf = sc._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.AbstractFileSystem.gs.impl",  "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
    hadoop_conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    hadoop_conf.set("fs.gs.auth.service.account.json.keyfile", credentials_location)
    hadoop_conf.set("fs.gs.auth.service.account.enable", "true")
    logging.info("spark context created")

    # Start Spark session using standalone cluster
    spark = SparkSession.builder \
        .config(conf=sc.getConf()) \
        .getOrCreate()
    logging.info("spark session created")

    # Gather Data
    df_raw_values = spark.read.parquet(f'gs://{gcs_bucket}/{gcs_raw_path}*.parquet')
    logging.info("raw data pulled from gcs")
    df_context = spark.read.parquet(f'gs://{gcs_bucket}/{gcs_sample_path}*.parquet')
    logging.info("sample context data pulled from gcs")
    # filter columns
    raman_cols = ["id"," 1-Raman spec recorded","2-PAT control(PAT_ref:PAT ref)","Fault flag"]
    sample_cols = ["id","Time (h)","Penicillin concentration(P:g/L)","Fault reference(Fault_ref:Fault ref)","0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)"]
    # split raw df into relevant sample values and raman measurement data
    df_samples = df_raw_values.select(sample_cols) \
        .withColumnRenamed("id","id_sample")
    df_raman = df_raw_values.select(raman_cols) \
        .withColumnRenamed("id","id_raman")
    logging.info("raman and sample spark dfs created")

    # find most recent existing record in T_SAMPLE_CONTEXT
    # gather all sample records produced in the last 5 minutes
    current_time = datetime.utcnow()
    if full_backfill is True:
        back_time = current_time - timedelta(days=2)
    else:
        back_time = current_time - timedelta(minutes=5)
    current_ts = current_time.strftime("%Y-%m-%d %H:%M:%S")
    back_ts = back_time.strftime("%Y-%m-%d %H:%M:%S")
    date_range = (current_ts,back_ts)
    where_clause = """sample_ts BETWEEN TO_TIMESTAMP('{1}','yyyy-MM-dd HH:mm:ss') AND TO_TIMESTAMP('{0}','yyyy-MM-dd HH:mm:ss')""".format(*date_range)
    try:
        most_recent_time = spark.read.format("bigquery") \
            .option("project",PROJECT_ID) \
            .option("dataset","test_schema") \
            .option("table","test_table") \
            .load() \
            .where(where_clause) \
            .agg(F.max("sample_ts")) \
            .collect()[0][0]
        logging.info(f"Some existing records found - starting after most recent existing time")
    except Exception as e:
        logging.info(f"No existing records found within 5 minutes of current time")
        most_recent_time = back_time
    most_recent_ts = most_recent_time.strftime("%Y-%m-%d %H:%M:%S")

    # gather new data that has not been traced into T_SAMPLE_CONTEXT
    date_range = (current_ts,most_recent_ts)
    df_context = df_context.select(["id","Batch Number","sample_ts"]) \
        .where(where_clause) \
        .withColumnRenamed("Batch Number","batch_number")
    # join to raman and sample dfs
    df_sample_context = df_samples.join(df_context,df_samples.id_sample == df_context.id,"inner").drop("id_sample")
    df_raman_context = df_raman.join(df_context,df_raman.id_raman == df_context.id,"inner").drop("id_raman")
    logging.info(f"Raman and sample value data joined to context data")

    # rename columns
    sample_colnames = ["time_hrs","penicillin_concentration_g_l","fault_reference","recipe_0_or_operator_1_controlled","id","batch_number","sample_ts"]
    raman_colnames = ["1_raman_spec_recorded","2_pat_control","fault_flag","id","batch_number","sample_ts"]
    df_sample_context = df_sample_context.toDF(*sample_colnames)
    df_raman_context = df_raman_context.toDF(*raman_colnames)
    # fill null values with 0
    df_sample_context = df_sample_context.fillna(0)
    df_raman_context = df_raman_context.fillna(0)

    # define schema for T_SAMPLE_CONTEXT
    sample_schema = T.StructType([
        T.StructField("time_hrs",T.DoubleType()),
        T.StructField("penicillin_concentration_g_l",T.DoubleType()),
        T.StructField("fault_reference",T.LongType()),
        T.StructField("recipe_0_or_operator_1_controlled",T.LongType()),
        T.StructField("id",T.IntegerType()),
        T.StructField("batch_number",T.LongType()),
        T.StructField("sample_ts",T.TimestampType())
    ])

    # define schema for T_RAMAN_CONTEXT
    raman_schema = T.StructType([
        T.StructField("id", T.IntegerType()),
        T.StructField("batch_number",T.LongType()),
        T.StructField("sample_ts",T.TimestampType()),
        T.StructField("1_raman_spec_recorded",T.LongType()),
        T.StructField("2_pat_control",T.LongType()),
        T.StructField("fault_flag",T.LongType())
    ])

    # add new sample context data to table
    logging.info(f"About to insert new data to T_SAMPLE_CONTEXT")
    df_sample_context.write.format("bigquery") \
        .option("temporaryGcsBucket", gcs_bucket) \
        .option("table", project_id+".test_schema.t_sample_context") \
        .option("createDisposition", "CREATE_IF_NEEDED") \
        .option("writeDisposition", "WRITE_TRUNCATE") \
        .option("schema", sample_schema.json()) \
        .mode("append") \
        .save()

    # add new raman context data to table
    logging.info(f"About to insert new data to T_RAMAN_CONTEXT")
    df_raman_context.write.format("bigquery") \
        .option("temporaryGcsBucket", gcs_bucket) \
        .option("table", project_id+".test_schema.t_raman_context") \
        .option("createDisposition", "CREATE_IF_NEEDED") \
        .option("writeDisposition", "WRITE_TRUNCATE") \
        .option("schema", raman_schema.json()) \
        .mode("append") \
        .save()