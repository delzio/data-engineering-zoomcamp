-- Create bigquery table from external table
create or replace table homework.green_trip_data_2022 as 
select * from `homework.external_green_trip_data_2022`;
