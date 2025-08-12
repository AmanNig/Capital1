#!/bin/bash

# Agricultural NLP Pipeline Installation Script
# This script installs all dependencies and sets up the pipeline

echo "🚜 Agricultural NLP Pipeline Installation"
echo "=========================================="

# Check if Python 3.8+ is installed
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version)"
else
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi

# Check if pip is installed
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 is installed"
else
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip3 install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Install spaCy English model
echo "🔤 Installing spaCy English model..."
python3 -m spacy download en_core_web_sm

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pip3 install -e .

# Check if installation was successful
echo "🧪 Testing installation..."
python3 -c "
from nlp_pipeline import QueryProcessingPipeline
pipeline = QueryProcessingPipeline()
result = pipeline.process_query('गेहूं की फसल में पानी कब देना चाहिए?')
print('✅ Pipeline test successful!')
print(f'Language: {result.primary_language}')
print(f'Intent: {result.primary_intent}')
print(f'Confidence: {result.intent_confidence:.3f}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "📚 Next steps:"
    echo "1. Run the demo: python3 demo.py"
    echo "2. Start the API: python3 api.py"
    echo "3. Run tests: python3 test_pipeline.py"
    echo ""
    echo "🌐 API will be available at: http://localhost:8000"
    echo "📖 API documentation: http://localhost:8000/docs"
    echo ""
    echo "📖 For more information, see README.md"
else
    echo "❌ Installation test failed. Please check the error messages above."
    exit 1
fi
