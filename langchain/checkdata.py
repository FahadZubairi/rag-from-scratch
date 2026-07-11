import pandas as pd
import sqlite3
df = pd.read_csv("results.csv")
conn = sqlite3.connect("football.db")
df.to_sql("matches",conn, if_exists="replace",index=False)
print(f"Database Created Successfully.")
print(f"Table 'matches' has {len(df)} rows")

cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM matches WHERE tournament = 'FIFA World Cup'")
print(f"World cup matches in database: {cursor.fetchone()[0]}")
conn.close()