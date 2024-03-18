--top 3 zones with most number of pickups
SELECT 
    taxi_zone.zone pickup_zone,
    count(1) trip_count
FROM trip_data
INNER JOIN taxi_zone
    ON trip_data.pulocationid = taxi_zone.location_id
WHERE tpep_pickup_datetime >= (
    select max(tpep_pickup_datetime) - INTERVAL '17 hours' from trip_data
)
GROUP BY
    taxi_zone.zone
ORDER BY trip_count desc
LIMIT 3;
--ANS=LaGuardia Airport, JFK Airport, Lincoln Square East