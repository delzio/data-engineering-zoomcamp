from datetime import datetime

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def camel_to_snake(col_name):
    snake_name = ""
    is_upper = col_name[0].isupper()

    for char in col_name:
        if char.isupper() and char.isupper() != is_upper:
            snake_name = snake_name + "_" + char.lower()
        else:
            snake_name = snake_name + char.lower()
        is_upper = char.isupper()
    
    return snake_name


@transformer
def transform(data, *args, **kwargs):
    print(len(set(data["lpep_pickup_datetime"].dt.date)))
    clean_data = data[(data["passenger_count"] != 0) & (data["trip_distance"] != 0)]
    print(len(set(clean_data["lpep_pickup_datetime"].dt.date)))
    clean_data["lpep_pickup_date"] = clean_data["lpep_pickup_datetime"].dt.date
    clean_data.columns = [camel_to_snake(col) for col in clean_data.columns]

    return clean_data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
