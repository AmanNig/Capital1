@echo off
echo ========================================
echo   Agricultural Advisor Bot Setup
echo ========================================
echo.
echo This script will set up everything needed to run the agricultural advisor bot.
echo It will run non-interactively and start the web app when done.
echo.

python setup_and_run.py --non-interactive --interface web

echo.
echo Setup finished. If the web app did not start, run: python run_streamlit.py
