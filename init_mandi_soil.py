import sqlite3
import pandas as pd

def init_db(price_csv, soil_csv, db_path="agri_data.db"):
    conn = sqlite3.connect(db_path)
    
    # Load prices CSV → store into table 'mandi_prices'
    df_price = pd.read_csv(price_csv)
    df_price.to_sql("mandi_prices", conn, if_exists="replace", index=False)
    
    # Load soil CSV → store into table 'soil_health'
    df_soil = pd.read_csv(soil_csv)
    df_soil.to_sql("soil_health", conn, if_exists="replace", index=False)
    
    conn.close()
    print("✅ Database initialized with prices and soil data")



if __name__ == "__main__":
    init_db(
        price_csv="/Users/manavjeetsingh/Desktop/CapitalOne/Capital1/mandi_prices.csv",
        soil_csv="/Users/manavjeetsingh/Desktop/CapitalOne/Capital1/soil_health.csv"
    )

