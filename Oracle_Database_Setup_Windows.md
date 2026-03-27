# Oracle Database Setup Guide (Windows)

## Step 1: Install Oracle Instant Client

1. Go to the Instant Client download page.
2. Download Instant Client Basic (the top package) for Windows x64.
3. Extract the ZIP to a folder such as `C:\oracle\instantclient`.
4. Add that folder to your `PATH`:
	a. Search "environment variables" in the Start menu.
	b. Under System Variables, find `Path` -> `Edit` -> `New` -> paste your path.
	c. Click `OK` and restart any open terminals.

## Step 2: Pick Your Client and Connect

### DBeaver (Easiest GUI - Free)

5. Download from https://dbeaver.io/download.
6. New Connection -> Oracle.
7. Enter your host, port (usually 1521), and either Service Name or SID.
8. Enter your username/password, then select Test Connection.
