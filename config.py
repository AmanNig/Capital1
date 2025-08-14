#!/usr/bin/env python3
"""
Configuration file for API keys and settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Weather API Configuration (if needed)
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# Database Configuration
DEFAULT_DB_DIR = "improved_vector_db"

# Model Configuration
DEFAULT_MODEL = "llama3-8b-8192"

def validate_config():
    """Validate that required configuration is present"""
    missing_keys = []
    
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    
    if missing_keys:
        print("⚠️  Missing required environment variables:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n💡 To set environment variables:")
        print("   PowerShell: $env:GROQ_API_KEY='your_api_key'")
        print("   Command Prompt: set GROQ_API_KEY=your_api_key")
        print("   Or create a .env file with: GROQ_API_KEY=your_api_key")
        return False
    
    return True
