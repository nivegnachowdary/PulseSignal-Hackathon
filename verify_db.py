import sqlite3

conn = sqlite3.connect("pulsesignal.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM structured_signals")

for row in cursor.fetchall():
    print(row)

conn.close()