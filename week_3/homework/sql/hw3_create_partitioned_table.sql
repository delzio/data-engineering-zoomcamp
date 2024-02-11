-- create partitioned table from non-partitioned table
create or replace table homework.partitioned_green_trip_data
partition by DATE(lpep_pickup_datetime)
cluster by pulocation_id as
select * from `homework.external_green_trip_data_2022`;
