import sqlite3
import os

# Define the SQLite3 database file name
PATH_DB='../Data/sqlite/'
db_file = 'candle_CW8.db'

# Define the SQL script file name
sql_schema_file = 'dump_maria_nodata_convert_231126.sql'  # Schema File
sql_data_file = 'dump_maria_data_convert_231126.sql'  # Data File
sql_candle_file = 'dump_maria_candle_231130.sql'  # Candle File

try:
    # Connect to the SQLite3 database (it will be created if it doesn't exist)
    conn = sqlite3.connect(os.path.join(PATH_DB,db_file))
    cursor = conn.cursor()

    # Read the SQL script from the file
    with open(os.path.join(PATH_DB,sql_candle_file), 'r') as script_file:
        sql_script = script_file.read()

    # Execute the SQL script
    cursor.executescript(sql_script)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Script {sql_candle_file} excuted successfully.")

except sqlite3.Error as e:
    print("SQLite error:", e)