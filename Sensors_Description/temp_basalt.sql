-- slope = "LEO West"
-- Physical quantity = Soil temperature
-- Physical symbol = T_soil
-- X-coordinate [m] = 2
-- Y-coordinate [m] = 10
-- Z-coordinate [m] = -0.05
-- Depth = 5 cm below surface
-- Workbook sheet: soilTemp
-- Workbook row: 2183
-- Exact source channel name: LEO-W_10_2_1_5TM
-- Oracle table: leo_west.datavalues
-- sensorid = 52
-- variableid = 3  -- soilTemp
-- Units = degC
-- Requested window:
-- start_date = "2014-Jan-01 00:00"
-- end_date   = "2014-Jan-01 23:59"

SELECT
    dv.localdatetime,
    dv.datavalue
FROM
    leo_west.datavalues dv
WHERE
    dv.sensorid = 52
    AND dv.variableid = 3
    AND dv.localdatetime >= TO_DATE('2026-03-25 00:00', 'YYYY-MM-DD HH24:MI')
    AND dv.localdatetime <= TO_DATE('2026-03-25 23:59', 'YYYY-MM-DD HH24:MI')
ORDER BY
    dv.localdatetime;
