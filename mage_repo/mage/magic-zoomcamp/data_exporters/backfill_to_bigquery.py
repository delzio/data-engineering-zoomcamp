import pyarrow as pa
import pyarrow.parquet as pq
from pandas import DataFrame
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/src/.keys/mage-gcp.json"
bucket_name = 'mage-zoomcamp-intricate-reef-411403'
project_id= 'intricate-reef-411403'

@data_exporter
def export_data(data, *args, **kwargs):
    
    now = kwargs.get('execution_date')
    execution_filepath = now.strftime("%Y-%m")

    bucket_name = 'mage-zoomcamp-intricate-reef-411403'
    root_path = f"{bucket_name}/week_4/green/{execution_filepath}"

    table = pa.Table.from_pandas(data)

    gcs = pa.fs.GcsFileSystem()

    pq.write_to_dataset(
        table,
        root_path=root_path,
        filesystem=gcs,
        use_deprecated_int96_timestamps=True
    )


