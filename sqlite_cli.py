import sqlite3
import pandas as pd
import sys

def sqlite_cli():
    """Simple SQLite command-line interface"""
    print("SQLite CLI for agri_data.db")
    print("Type 'quit' or 'exit' to exit")
    print("Type 'help' for available commands")
    print("-" * 40)
    
    conn = sqlite3.connect('agri_data.db')
    
    while True:
        try:
            # Get user input
            query = input("\nsqlite> ").strip()
            
            # Handle special commands
            if query.lower() in ['quit', 'exit']:
                break
            elif query.lower() == 'help':
                print_help()
                continue
            elif query.lower() == 'tables':
                show_tables(conn)
                continue
            elif query.lower() == 'schema':
                show_schema(conn)
                continue
            elif not query:
                continue
            
            # Execute SQL query
            try:
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    print(f"\nQuery returned {len(df)} rows:")
                    print(df.to_string(index=False))
                else:
                    print("Query executed successfully. No data returned.")
                    
            except Exception as e:
                print(f"Error executing query: {e}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break
    
    conn.close()
    print("Goodbye!")

def print_help():
    """Print help information"""
    print("\nAvailable commands:")
    print("  tables    - Show all tables")
    print("  schema    - Show table schemas")
    print("  help      - Show this help")
    print("  quit/exit - Exit the program")
    print("\nOr enter any SQL query directly.")
    print("\nExample queries:")
    print("  SELECT * FROM mandi_prices LIMIT 5;")
    print("  SELECT DISTINCT Commodity FROM mandi_prices;")
    print("  SELECT * FROM soil_health;")

def show_tables(conn):
    """Show all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("\nTables in database:")
    for table in tables:
        print(f"  - {table[0]}")

def show_schema(conn):
    """Show schema for all tables"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\nSchema for table '{table_name}':")
        print("-" * 30)
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for col in columns:
            print(f"  {col[1]} ({col[2]})")

if __name__ == "__main__":
    sqlite_cli()
