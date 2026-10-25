-- Store engine measurements from all four NASA C-MAPSS datasets.
CREATE TABLE IF NOT EXISTS engine_cycles (
    dataset_group VARCHAR(5) NOT NULL,
    dataset_split VARCHAR(5) NOT NULL,
    unit_number INTEGER NOT NULL,
    time_in_cycles INTEGER NOT NULL,

    -- Engine operating conditions.
    operational_settings1 DOUBLE PRECISION,
    operational_settings2 DOUBLE PRECISION,
    operational_settings3 DOUBLE PRECISION,

    -- Measurements collected from the 21 sensors.
    sensor_measurement1 DOUBLE PRECISION,
    sensor_measurement2 DOUBLE PRECISION,
    sensor_measurement3 DOUBLE PRECISION,
    sensor_measurement4 DOUBLE PRECISION,
    sensor_measurement5 DOUBLE PRECISION,
    sensor_measurement6 DOUBLE PRECISION,
    sensor_measurement7 DOUBLE PRECISION,
    sensor_measurement8 DOUBLE PRECISION,
    sensor_measurement9 DOUBLE PRECISION,
    sensor_measurement10 DOUBLE PRECISION,
    sensor_measurement11 DOUBLE PRECISION,
    sensor_measurement12 DOUBLE PRECISION,
    sensor_measurement13 DOUBLE PRECISION,
    sensor_measurement14 DOUBLE PRECISION,
    sensor_measurement15 DOUBLE PRECISION,
    sensor_measurement16 DOUBLE PRECISION,
    sensor_measurement17 DOUBLE PRECISION,
    sensor_measurement18 DOUBLE PRECISION,
    sensor_measurement19 DOUBLE PRECISION,
    sensor_measurement20 DOUBLE PRECISION,
    sensor_measurement21 DOUBLE PRECISION,

    -- Remaining useful life, calculated by our ETL pipeline.
    rul INTEGER NOT NULL,

    -- Prevent duplicate engine-cycle records.
    PRIMARY KEY (dataset_group, dataset_split, unit_number, time_in_cycles),

    CHECK (dataset_group IN ('FD001', 'FD002', 'FD003', 'FD004')),
    CHECK (dataset_split IN ('train', 'test')),
    CHECK (unit_number > 0),
    CHECK (time_in_cycles > 0),
    CHECK (rul >= 0)
);
