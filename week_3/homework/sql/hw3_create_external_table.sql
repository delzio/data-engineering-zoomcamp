-- create external table from bucket parquet files
create or replace external table homework.external_green_trip_data_2022 (
  vendor_id INT64, 
  lpep_pickup_datetime DATETIME,
  lpep_dropoff_datetime DATETIME,
  store_and_fwd_flag STRING,
  ratecode_id INT64, 
  pulocation_id INT64, 
  dolocation_id INT64,
  passenger_count INT64, 
  trip_distance FLOAT64, 
  fare_amount FLOAT64, 
  extra FLOAT64, 
  mta_tax FLOAT64,
  tip_amount FLOAT64, 
  tolls_amount FLOAT64, 
  ehail_fee FLOAT64, 
  improvement_surcharge FLOAT64,
  total_amount FLOAT64, 
  payment_type INT64, 
  trip_type INT64, 
  congestion_surcharge FLOAT64
)
options (
  format = "PARQUET",
  uris = ['gs://mage-zoomcamp-intricate-reef-411403/homework3/*']
);
