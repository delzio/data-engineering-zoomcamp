
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
    data.columns = [camel_to_snake(col) for col in data.columns]
    print(data.columns)
    print(type(data["lpep_pickup_datetime"][0]))

    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
