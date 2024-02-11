-- Select queries
select count(distinct pulocation_id) from `homework.external_green_trip_data_2022`;

select count(distinct fare_amount) from `homework.green_trip_data_2022`;

select count(1) from `homework.external_green_trip_data_2022` where fare_amount = 0;

select count(distinct pulocation_id) from `homework.green_trip_data_2022` 
where lpep_pickup_datetime >= TIMESTAMP('2022-06-01') and lpep_pickup_datetime <= TIMESTAMP('2022-06-30');

select count(distinct pulocation_id) from `homework.partitioned_green_trip_data` 
where lpep_pickup_datetime >= TIMESTAMP('2022-06-01') and lpep_pickup_datetime <= TIMESTAMP('2022-06-30');
