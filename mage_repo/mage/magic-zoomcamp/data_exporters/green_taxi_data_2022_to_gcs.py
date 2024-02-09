from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.google_cloud_storage import GoogleCloudStorage
from pandas import DataFrame
from os import path

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    
    now = kwargs.get('execution_date')
    execution_filepath = now.strftime("%Y-%m")

    config_path = path.join(get_repo_path(), 'io_config.yaml')
    config_profile = 'dev'

    bucket_name = 'mage-zoomcamp-intricate-reef-411403'
    object_key = f'homework3/{execution_filepath}/montly_trips.parquet'

    GoogleCloudStorage.with_config(ConfigFileLoader(config_path, config_profile)).export(
        data,
        bucket_name,
        object_key,
    )



