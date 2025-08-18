#!/usr/bin/env python3
"""
Agricultural Advisor Bot - Complete Setup and Run Script
This script handles the complete setup process:
1. Install all dependencies
2. Create and initialize databases
3. Process policy documents
4. Start the bot (CLI or Web interface)
"""

import os
import sys
import subprocess
import time
import json
import sqlite3
from pathlib import Path
import platform
import argparse

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_step(step_num, text):
    """Print a formatted step"""
    print(f"{Colors.OKBLUE}[STEP {step_num}] {text}{Colors.ENDC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install all required dependencies"""
    print_step(2, "Installing dependencies...")
    
    try:
        # Install base requirements
        print_info("Installing base requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print_success("Base requirements installed")
        
        # Install Streamlit requirements
        print_info("Installing Streamlit requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"], 
                      check=True, capture_output=True, text=True)
        print_success("Streamlit requirements installed")
        
        # Install spaCy model
        print_info("Installing spaCy English model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                      check=True, capture_output=True, text=True)
        print_success("spaCy model installed")
        
        # Install NLTK data
        print_info("Installing NLTK data...")
        subprocess.run([sys.executable, "-c", "import nltk; nltk.download('wordnet'); nltk.download('punkt')"], 
                      check=True, capture_output=True, text=True)
        print_success("NLTK data installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        print_info("You can try installing manually:")
        print_info("pip install -r requirements.txt")
        print_info("pip install -r requirements_streamlit.txt")
        return False

def create_directories():
    """Create necessary directories"""
    print_step(3, "Creating directories...")
    
    directories = [
        "models",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print_success(f"Created directory: {directory}")
    
    return True

def ensure_env_file():
    """Create a .env template if missing (non-blocking)."""
    try:
        env_path = Path(".env")
        if not env_path.exists():
            print_step(3, "Creating .env template (optional)...")
            content = (
                "# Environment variables for Agricultural Advisor Bot\n"
                "# Add your API keys if available. The app works without them, but with reduced features.\n"
                "GROQ_API_KEY=\n"
                "WEATHER_API_KEY=\n"
            )
            env_path.write_text(content, encoding="utf-8")
            print_success("Created .env template (edit to add API keys as needed)")
        else:
            print_info(".env file found (skipping)")
        return True
    except Exception as e:
        print_warning(f"Could not create .env file: {e}")
        return False

def initialize_database():
    """Initialize the agricultural database"""
    print_step(4, "Initializing database...")
    
    try:
        if os.path.exists("agri_data.db"):
            print_info("Database already exists, skipping initialization")
            return True
        
        print_info("Running database initialization script...")
        result = subprocess.run([sys.executable, "init_mandi_soil.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("Database initialized successfully")
            return True
        else:
            print_error(f"Database initialization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Error initializing database: {e}")
        return False

def process_policy_documents():
    """Process policy documents for the improved vector database"""
    print_step(5, "Processing policy documents...")
    
    try:
        if os.path.exists("improved_vector_db/metadata.json"):
            print_info("Policy database already exists, skipping processing")
            return True
        
        # Run PDF vector processor first
        print_info("Running PDF vector processor...")
        pdf_result = subprocess.run(
            [sys.executable, "pdf_vector_processor.py", "pdfs"],
            capture_output=True,
            text=True
        )
        if pdf_result.returncode != 0:
            print_error("PDF vector processing failed")
            if pdf_result.stdout:
                print_info(pdf_result.stdout)
            if pdf_result.stderr:
                print_error(pdf_result.stderr)
            return False
        else:
            print_success("PDF vector processor completed successfully")

        # Use the improved policy chatbot to build the improved vector DB
        print_info("Running improved policy document processor...")
        result = subprocess.run(
            [sys.executable, "improved_policy_chatbot.py", "--build", "pdfs"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print_success("Policy documents processed successfully with ImprovedPolicyChatbot")
            return True
        else:
            # Surface stdout as well to aid debugging
            print_error("Policy processing failed with ImprovedPolicyChatbot")
            if result.stdout:
                print_info(result.stdout)
            if result.stderr:
                print_error(result.stderr)
            return False
            
    except Exception as e:
        print_error(f"Error processing policy documents: {e}")

def verify_setup():
    """Verify that all components are properly set up"""
    print_step(6, "Verifying setup...")
    
    checks = []
    
    # Check database
    if os.path.exists("agri_data.db"):
        try:
            conn = sqlite3.connect("agri_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM mandi_prices")
            price_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM soil_health")
            soil_count = cursor.fetchone()[0]
            conn.close()
            
            checks.append(f"Database: {price_count:,} price records, {soil_count:,} soil records")
        except Exception as e:
            checks.append(f"Database error: {e}")
    else:
        checks.append("Database not found")
    
    # Check policy database
    if os.path.exists("improved_vector_db/metadata.json"):
        try:
            with open("improved_vector_db/metadata.json", "r") as f:
                metadata = json.load(f)
            # Support both legacy and improved metadata keys
            num_sections = metadata.get('num_sections', metadata.get('total_sections', 0))
            num_documents = metadata.get('num_documents', None)
            if num_documents is not None:
                checks.append(f"Policy DB: {num_sections} sections across {num_documents} documents")
            else:
                checks.append(f"Policy DB: {num_sections} sections")
        except Exception as e:
            checks.append(f"Policy DB error: {e}")
    else:
        checks.append("Policy database not found")
    
    # Check PDFs
    pdf_count = len([f for f in os.listdir("pdfs") if f.endswith(".pdf")])
    checks.append(f"PDFs: {pdf_count} policy documents")
    
    # Check dependencies
    try:
        import torch
        import transformers
        import spacy
        import streamlit
        import plotly
        checks.append("Dependencies: All packages installed")
    except ImportError as e:
        checks.append(f"Dependencies: {e}")
    
    # Print results
    for check in checks:
        if "error" in check.lower() or "not found" in check.lower():
            print_warning(check)
        else:
            print_success(check)
    
    return all("error" not in check.lower() and "not found" not in check.lower() for check in checks)

def get_user_choice():
    """Get user choice for interface"""
    print_header("Choose Interface")
    print("1. Command Line Interface (CLI)")
    print("2. Web Interface (Streamlit)")
    print("3. Exit")
    
    while True:
        try:
            choice = input(f"\n{Colors.OKCYAN}Enter your choice (1-3): {Colors.ENDC}").strip()
            if choice in ['1', '2', '3']:
                return choice
            else:
                print_error("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

def start_cli_bot():
    """Start the CLI bot"""
    print_header("Starting CLI Bot")
    print_info("Starting interactive agricultural advisor bot...")
    print_info("Press Ctrl+C to exit")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "agricultural_advisor_bot.py", "--interactive"])
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print_error(f"Error starting CLI bot: {e}")

def start_web_interface():
    """Start the Streamlit web interface"""
    print_header("Starting Web Interface")
    print_info("Starting Streamlit web interface...")
    print_info("The web interface will be available at: http://localhost:8501")
    print_info("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\nWeb interface stopped by user")
    except Exception as e:
        print_error(f"Error starting web interface: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Agricultural Advisor Bot - Complete Setup and Runner")
    parser.add_argument("--interface", choices=["cli", "web", "none"], default="web",
                        help="What to start after setup (default: web)")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Run end-to-end without any prompts (auto-selects interface)")
    args = parser.parse_args()

    print_header("Agricultural Advisor Bot - Complete Setup")
    print_info("This script will set up everything needed to run the agricultural advisor bot")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create .env template if missing (optional)
    ensure_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print_error("Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print_error("Failed to create directories.")
        sys.exit(1)
    
    # Initialize database
    if not initialize_database():
        print_error("Failed to initialize database.")
        sys.exit(1)
    
    # Process policy documents
    if not process_policy_documents():
        print_error("Failed to process policy documents.")
        sys.exit(1)
    
    # Verify setup
    if not verify_setup():
        print_warning("Some components may not be properly set up.")
        print_info("You can continue, but some features may not work correctly.")
    
    print_header("Setup Complete!")
    print_success("All components have been set up successfully!")
    
    # Non-interactive or explicit interface selection
    if args.non_interactive or args.interface in ("cli", "web", "none"):
        if args.interface == "cli":
            start_cli_bot()
        elif args.interface == "web":
            start_web_interface()
        else:
            print_info("Setup finished. Not launching any interface (--interface none)")
            sys.exit(0)
        return
    
    # Interactive choice (fallback)
    choice = get_user_choice()
    if choice == '1':
        start_cli_bot()
    elif choice == '2':
        start_web_interface()
    else:
        print_info("Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
