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

def get_latest_price(city, crop, db_path="agri_data.db"):
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT Market, Commodity, Variety, Arrival_Date, Modal_Price
    FROM mandi_prices
    WHERE District LIKE ? AND Commodity LIKE ?
    ORDER BY date(Arrival_Date) DESC
    LIMIT 1;
    """
    result = conn.execute(query, (f"%{city}%", f"%{crop}%")).fetchone()
    conn.close()
    
    if result:
        market, commodity, variety, date, price = result
        return f"Latest {commodity} ({variety}) price in {city} ({market}) on {date}: ₹{price}/quintal"
    else:
        return f"No price data found for {crop} in {city}."

def get_soil_health(city, db_path="agri_data.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM soil_health WHERE District LIKE ? LIMIT 1;"
    result = conn.execute(query, (f"%{city}%",)).fetchone()
    columns = [col[0] for col in conn.execute("PRAGMA table_info(soil_health);")]
    conn.close()
    
    if result:
        soil_data = dict(zip(columns, result))
        return f"Soil health for {city}: pH {soil_data['pH']}, Organic Carbon {soil_data['OC (%)']}%, Nitrogen {soil_data['N (kg/ha)']} kg/ha, Phosphorus {soil_data['P (kg/ha)']} kg/ha, Potassium {soil_data['K (kg/ha)']} kg/ha"
    else:
        return f"No soil health data found for {city}."



if __name__ == "__main__":
    init_db(
        price_csv="/Users/manavjeetsingh/Desktop/CapitalOne/Capital1/mandi_prices.csv",
        soil_csv="/Users/manavjeetsingh/Desktop/CapitalOne/Capital1/soil_health.csv"
    )

