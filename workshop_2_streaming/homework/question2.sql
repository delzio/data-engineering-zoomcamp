-- drop the original view and recreate
DROP VIEW zone_pair_stats;

-- recreate the view with trip counts
CREATE MATERIALIZED VIEW zone_pair_stats as
SELECT
    pickup_zone.zone pickup_zone, 
    dropoff_zone.zone dropoff_zone,
    count(1) trip_count,
    max(tpep_dropoff_datetime - tpep_pickup_datetime) max_trip_time,
    avg(tpep_dropoff_datetime - tpep_pickup_datetime) avg_trip_time,
    min(tpep_dropoff_datetime - tpep_pickup_datetime) min_trip_time
FROM trip_data
INNER JOIN taxi_zone as pickup_zone
    ON trip_data.pulocationid = pickup_zone.location_id
INNER JOIN taxi_zone as dropoff_zone
    ON trip_data.dolocationid = dropoff_zone.location_id
GROUP BY
    pickup_zone.zone, dropoff_zone.zone;


-- select the trip count for zone pair with max avg duration
SELECT trip_count
FROM zone_pair_stats
WHERE avg_trip_time = (
    SELECT max(avg_trip_time)
    FROM zone_pair_stats
);
--ANS=1

