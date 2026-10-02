PYTHON ?= python3
ORACLE_LIBRARY_PATH ?= /opt/oracle/instantclient_19_26:/tmp/ora_compat

.PHONY: co2-refresh co2-status
co2-refresh:
	LD_LIBRARY_PATH="$(ORACLE_LIBRARY_PATH):$$LD_LIBRARY_PATH" $(PYTHON) -B Project_description/sensorDB/co2_refresh.py refresh

co2-status:
	$(PYTHON) -B Project_description/sensorDB/co2_refresh.py status
