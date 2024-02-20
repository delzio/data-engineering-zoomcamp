import pyarrow as pa
import pyarrow.parquet as pq
from pandas import DataFrame
import os
from os import path
from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.google_cloud_storage import GoogleCloudStorage

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/src/.keys/mage-gcp.json"
bucket_name = 'mage-zoomcamp-intricate-reef-411403'
project_id= 'intricate-reef-411403'

@data_exporter
def export_data(data, *args, **kwargs):
    
    now = kwargs.get('execution_date')
    execution_filepath = now.strftime("%Y-%m")

    config_path = path.join(get_repo_path(), 'io_config.yaml')
    config_profile = 'dev'

    bucket_name = 'mage-zoomcamp-intricate-reef-411403'
    object_key = f'week_4/fhv/{execution_filepath}/fhv_monthly_trips.csv'

    GoogleCloudStorage.with_config(ConfigFileLoader(config_path, config_profile)).export(
        data,
        bucket_name,
        object_key,
    )


