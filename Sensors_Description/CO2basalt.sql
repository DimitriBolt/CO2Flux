SELECT
    dv.localdatetime,
    dv.datavalue
FROM
    leo_west.datavalues dv
WHERE
        dv.sensorid = 1000
    AND dv.variableid = 9
    AND dv.localdatetime >= DATE '2026-03-26'
ORDER BY
    dv.localdatetime DESC
FETCH FIRST 100 ROWS ONLY