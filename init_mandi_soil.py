import sqlite3
import pandas as pd
import os
import logging
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriculturalDataManager:
    """Manages agricultural data including mandi prices and soil health"""
    
    def __init__(self, db_path: str = "agri_data.db"):
        self.db_path = db_path
        self.conn = None
        
    def init_db(self, price_csv: str = "mandi_prices.csv", soil_csv: str = "soil_health.csv") -> bool:
        """Initialize database with prices and soil data"""
        try:
            # Check if CSV files exist
            if not os.path.exists(price_csv):
                logger.error(f"Price CSV file not found: {price_csv}")
                return False
                
            if not os.path.exists(soil_csv):
                logger.error(f"Soil CSV file not found: {soil_csv}")
                return False
            
            self.conn = sqlite3.connect(self.db_path)
            
            # Load prices CSV → store into table 'mandi_prices'
            logger.info(f"Loading price data from {price_csv}")
            df_price = pd.read_csv(price_csv)
            df_price.to_sql("mandi_prices", self.conn, if_exists="replace", index=False)
            logger.info(f"Loaded {len(df_price)} price records")
            
            # Load soil CSV → store into table 'soil_health'
            logger.info(f"Loading soil data from {soil_csv}")
            df_soil = pd.read_csv(soil_csv)
            df_soil.to_sql("soil_health", self.conn, if_exists="replace", index=False)
            logger.info(f"Loaded {len(df_soil)} soil records")
            
            # Create indexes for better performance
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mandi_district ON mandi_prices(District)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mandi_commodity ON mandi_prices(Commodity)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_soil_district ON soil_health(District)")
            
            self.conn.commit()
            self.conn.close()
            
            print("✅ Database initialized with prices and soil data")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            if self.conn:
                self.conn.close()
            return False
    
    def get_latest_price(self, city: str, crop: str) -> str:
        """Get latest price for a crop in a city"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT Market, Commodity, Variety, Arrival_Date, Modal_Price, Min_Price, Max_Price
            FROM mandi_prices
            WHERE District LIKE ? AND Commodity LIKE ?
            ORDER BY date(Arrival_Date) DESC
            LIMIT 1;
            """
            result = conn.execute(query, (f"%{city}%", f"%{crop}%")).fetchone()
            conn.close()
            
            if result:
                market, commodity, variety, date, modal_price, min_price, max_price = result
                return {
                    "market": market,
                    "commodity": commodity,
                    "variety": variety,
                    "date": date,
                    "modal_price": modal_price,
                    "min_price": min_price,
                    "max_price": max_price,
                    "formatted": f"Latest {commodity} ({variety}) price in {city} ({market}) on {date}: ₹{modal_price}/quintal (Range: ₹{min_price}-₹{max_price})",
                    "source": f"mandi_prices.csv → agri_data.db → mandi_prices table"
                }
            else:
                return {"error": f"No price data found for {crop} in {city}."}
                
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return {"error": f"Error retrieving price data: {str(e)}"}
    
    def get_soil_health(self, city: str) -> Dict[str, Any]:
        """Get soil health data for a city"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # First try exact match
            query = "SELECT * FROM soil_health WHERE District = ? LIMIT 1;"
            result = conn.execute(query, (city,)).fetchone()
            
            # If no exact match, try partial match
            if not result:
                query = "SELECT * FROM soil_health WHERE District LIKE ? LIMIT 1;"
                result = conn.execute(query, (f"%{city}%",)).fetchone()
            
            # If still no match, try reverse partial match (city name in district)
            if not result:
                query = "SELECT * FROM soil_health WHERE ? LIKE '%' || District || '%' LIMIT 1;"
                result = conn.execute(query, (city,)).fetchone()
            
            # Get column names
            columns = [col[0] for col in conn.execute("PRAGMA table_info(soil_health);")]
            conn.close()
            
            if result:
                # Create dictionary with column names as keys
                soil_data = {}
                for i, col in enumerate(columns):
                    soil_data[col] = result[i]
                
                # Debug: Print the actual data
                logger.info(f"Soil data found: {soil_data}")
                
                return {
                    "district": soil_data.get('District', city),
                    "ph": soil_data.get('pH', 0),
                    "organic_carbon": soil_data.get('OC (%)', 0),
                    "nitrogen": soil_data.get('N (kg/ha)', 0),
                    "phosphorus": soil_data.get('P (kg/ha)', 0),
                    "potassium": soil_data.get('K (kg/ha)', 0),
                    "zinc": soil_data.get('Zn (%)', 0),
                    "iron": soil_data.get('Fe (%)', 0),
                    "copper": soil_data.get('Cu (%)', 0),
                    "manganese": soil_data.get('Mn (%)', 0),
                    "boron": soil_data.get('B (%)', 0),
                    "sulfur": soil_data.get('S (%)', 0),
                    "formatted": f"Soil health for {city}: pH {soil_data.get('pH', 'N/A')}, Organic Carbon {soil_data.get('OC (%)', 'N/A')}%, Nitrogen {soil_data.get('N (kg/ha)', 'N/A')} kg/ha, Phosphorus {soil_data.get('P (kg/ha)', 'N/A')} kg/ha, Potassium {soil_data.get('K (kg/ha)', 'N/A')} kg/ha",
                    "source": f"soil_health.csv → agri_data.db → soil_health table"
                }
            else:
                # Debug: Let's see what's actually in the database
                conn = sqlite3.connect(self.db_path)
                debug_query = "SELECT District FROM soil_health;"
                debug_results = conn.execute(debug_query).fetchall()
                conn.close()
                available_districts = [row[0] for row in debug_results]
                return {"error": f"No soil health data found for '{city}'. Available districts: {available_districts}"}
                
        except Exception as e:
            logger.error(f"Error getting soil data: {e}")
            return {"error": f"Error retrieving soil data: {str(e)}"}
    
    def get_price_trends(self, city: str, crop: str, days: int = 30) -> Dict[str, Any]:
        """Get price trends for a crop in a city over specified days"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT Arrival_Date, Modal_Price, Min_Price, Max_Price
            FROM mandi_prices
            WHERE District LIKE ? AND Commodity LIKE ?
            ORDER BY date(Arrival_Date) DESC
            LIMIT ?;
            """
            results = conn.execute(query, (f"%{city}%", f"%{crop}%", days)).fetchall()
            conn.close()
            
            if results:
                prices = []
                for date, modal, min_price, max_price in results:
                    prices.append({
                        "date": date,
                        "modal_price": modal,
                        "min_price": min_price,
                        "max_price": max_price
                    })
                
                # Calculate trends
                if len(prices) > 1:
                    latest_price = prices[0]["modal_price"]
                    oldest_price = prices[-1]["modal_price"]
                    price_change = latest_price - oldest_price
                    price_change_percent = (price_change / oldest_price) * 100 if oldest_price > 0 else 0
                    
                    return {
                        "prices": prices,
                        "trend": "increasing" if price_change > 0 else "decreasing" if price_change < 0 else "stable",
                        "change": price_change,
                        "change_percent": price_change_percent,
                        "latest_price": latest_price,
                        "formatted": f"Price trend for {crop} in {city}: {'↗️ Increasing' if price_change > 0 else '↘️ Decreasing' if price_change < 0 else '→ Stable'} ({price_change_percent:.1f}% change)"
                    }
                else:
                    return {
                        "prices": prices,
                        "trend": "insufficient_data",
                        "formatted": f"Only one price record found for {crop} in {city}"
                    }
            else:
                return {"error": f"No price trend data found for {crop} in {city}."}
                
        except Exception as e:
            logger.error(f"Error getting price trends: {e}")
            return {"error": f"Error retrieving price trends: {str(e)}"}
    
    def get_available_crops(self, city: str) -> List[str]:
        """Get list of available crops for a city"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT DISTINCT Commodity
            FROM mandi_prices
            WHERE District LIKE ?
            ORDER BY Commodity;
            """
            results = conn.execute(query, (f"%{city}%",)).fetchall()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"Error getting available crops: {e}")
            return []
    
    def get_available_cities(self) -> List[str]:
        """Get list of available cities"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT DISTINCT District FROM mandi_prices ORDER BY District;"
            results = conn.execute(query).fetchall()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"Error getting available cities: {e}")
            return []

def init_db(price_csv: str = "mandi_prices.csv", soil_csv: str = "soil_health.csv", db_path: str = "agri_data.db") -> bool:
    """Initialize database with prices and soil data"""
    manager = AgriculturalDataManager(db_path)
    return manager.init_db(price_csv, soil_csv)

def get_latest_price(city: str, crop: str, db_path: str = "agri_data.db") -> str:
    """Get latest price for a crop in a city"""
    manager = AgriculturalDataManager(db_path)
    result = manager.get_latest_price(city, crop)
    return result.get("formatted", result.get("error", "Unknown error"))

def get_soil_health(city: str, db_path: str = "agri_data.db") -> str:
    """Get soil health data for a city"""
    manager = AgriculturalDataManager(db_path)
    result = manager.get_soil_health(city)
    return result.get("formatted", result.get("error", "Unknown error"))

if __name__ == "__main__":
    # Use relative paths for the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    price_csv = os.path.join(current_dir, "mandi_prices.csv")
    soil_csv = os.path.join(current_dir, "soil_health.csv")
    
    print("🔧 Initializing Agricultural Database...")
    print(f"📄 Price CSV: {price_csv}")
    print(f"🌱 Soil CSV: {soil_csv}")
    
    success = init_db(price_csv, soil_csv)
    
    if success:
        print("\n✅ Database initialization completed!")
        
        # Test the database
        manager = AgriculturalDataManager()
        
        # Test price data
        print("\n🧪 Testing price data...")
        test_cities = ["Mumbai", "Delhi", "Kanpur"]
        test_crops = ["Rice", "Wheat", "Cotton"]
        
        for city in test_cities[:1]:  # Test with first city
            for crop in test_crops[:1]:  # Test with first crop
                price_result = manager.get_latest_price(city, crop)
                if "error" not in price_result:
                    print(f"✅ {price_result['formatted']}")
                else:
                    print(f"❌ {price_result['error']}")
        
        # Test soil data
        print("\n🧪 Testing soil data...")
        for city in test_cities[:1]:
            soil_result = manager.get_soil_health(city)
            if "error" not in soil_result:
                print(f"✅ {soil_result['formatted']}")
            else:
                print(f"❌ {soil_result['error']}")
        
        # Show available data
        print("\n📊 Available Data Summary:")
        cities = manager.get_available_cities()
        print(f"   Cities: {len(cities)} available")
        if cities:
            print(f"   Sample cities: {', '.join(cities[:5])}")
        
        crops = manager.get_available_crops(cities[0] if cities else "")
        print(f"   Crops in {cities[0] if cities else 'N/A'}: {len(crops)} available")
        if crops:
            print(f"   Sample crops: {', '.join(crops[:5])}")
        
    else:
        print("❌ Database initialization failed!")

