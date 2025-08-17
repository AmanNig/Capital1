#!/usr/bin/env python3
"""
Streamlit Agricultural Advisor Bot Runner
This script runs the Streamlit web application for the Agricultural Advisor Bot.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies using: pip install -r requirements_streamlit.txt")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'agri_data.db',
        'mandi_prices.csv',
        'soil_health.csv',
        'improved_vector_db/metadata.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run the setup scripts first:")
        print("1. python init_mandi_soil.py")
        print("2. python pdf_vector_processor.py")
        return False
    
    print("✅ All data files found")
    return True

def run_streamlit():
    """Run the Streamlit application"""
    try:
        # Set environment variables for better performance
        env = os.environ.copy()
        env['STREAMLIT_SERVER_PORT'] = '8501'
        env['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
        env['STREAMLIT_SERVER_HEADLESS'] = 'true'
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Run Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ]
        
        print("🚀 Starting Streamlit Agricultural Advisor Bot...")
        print("📱 Web interface will be available at: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    """Main function"""
    print("🌾 Agricultural Advisor Bot - Streamlit Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main()
