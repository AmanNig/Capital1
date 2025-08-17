#!/bin/bash

echo "========================================"
echo "  Agricultural Advisor Bot Setup"
echo "========================================"
echo ""
echo "This script will set up everything needed to run the agricultural advisor bot."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Make the script executable
chmod +x setup_and_run.py

# Run the setup script
python3 setup_and_run.py

echo ""
echo "Setup completed!"
