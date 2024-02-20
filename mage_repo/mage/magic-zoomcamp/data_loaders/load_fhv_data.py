import io
import pandas as pd
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    
    #execution_date = kwargs.get('execution_date').date()
    #execution_month = execution_date.strftime("%Y-%m")

    execution_date = kwargs.get('execution_date').date()
    execution_month = execution_date.strftime("%Y-%m")

    chunk_size = 100000
    chunks = []
    
    taxi_dtypes = {
                    'dispatching_base_num': str,
                    'PULocationID':pd.Int64Dtype(),
                    'DOLocationID':pd.Int64Dtype(),
                    'SR_Flag':str,
                    'Affiliated_base_number':str
                }
    
    parse_dates = ['pickup_datetime', 'dropOff_datetime']

    url = f"https://github.com/DataTalksClub/nyc-tlc-data/releases/download/fhv/fhv_tripdata_{execution_month}.csv.gz"

    for chunk in pd.read_csv(url, sep=',', compression="gzip", chunksize=chunk_size, dtype=taxi_dtypes, parse_dates=parse_dates):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
