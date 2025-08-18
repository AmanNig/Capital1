import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from agricultural_advisor_bot import AgriculturalAdvisorBot
    from weather_service import WeatherService
    from improved_policy_chatbot import ImprovedPolicyChatbot
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please ensure all required files are in the same directory")

# Page configuration
st.set_page_config(
    page_title="Agricultural Advisor Bot",
    page_icon="🌾",
    layout="wide"
)

# Clean CSS styling

st.markdown("""
<style>
    .stApp {
        background: url('https://www.agora-agriculture.org/fileadmin/Sections/Home/AGRAR_Website_Verlauf_20240924_Landinpage-Header_1920x1080.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background: rgba(255,255,255,0.85);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }
    .header-section {
        background: rgba(255,255,255,0.85);
        color: #111;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .info-section {
        background: rgba(255,255,255,0.85);
        color: #111;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: rgba(255,255,255,0.85);
        color: #111;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        max-width: 80%;
        color: #111;
        background: rgba(255,255,255,0.6);
    }
    .user-message {
        background: rgba(227,242,253,0.7);
        border-left: 4px solid #2196f3;
        margin-left: auto;
        color: #111;
    }
    .bot-message {
        background: rgba(241,248,233,0.7);
        border-left: 4px solid #4caf50;
        color: #111;
    }
    .input-area {
        background: rgba(255,255,255,0.85);
        color: #111;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .feature-card {
        background: rgba(255,255,255,0.6);
        color: #111;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: bold;
    }
    .status-online {
        background: #d4edda;
        color: #155724;
    }
    .status-offline {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_bot():
    """Initialize the agricultural advisor bot with default settings"""
    try:
        if st.session_state.bot is None:
            bot = AgriculturalAdvisorBot()
            # Set default values to avoid initialization issues
            bot.user_city = "Kanpur"  # Default city
            bot.user_crop = "Wheat"   # Default crop
            bot.user_language = "English"  # Default language
            bot.is_initialized = True  # Mark as initialized
            st.session_state.bot = bot
        return True
    except Exception as e:
        st.error(f"Error initializing bot: {e}")
        return False

def process_query_with_fallback(query: str) -> str:
    """Process query with fallback handling for setup issues"""
    try:
        if not st.session_state.bot:
            return "❌ Bot not initialized. Please start the bot first."
        
        # Try to process the query normally
        response = st.session_state.bot.process_query(query)
        
        # Check if response indicates setup is required
        if "Setup Required" in response or "complete the initial setup" in response:
            # Extract information from the query and set it automatically
            query_lower = query.lower()
            
            # Try to extract city from query
            cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'kanpur', 'lucknow', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik']
            for city in cities:
                if city in query_lower:
                    st.session_state.bot.user_city = city.title()
                    break
            
            # Try to extract crop from query
            crops = ['wheat', 'rice', 'corn', 'maize', 'sugarcane', 'cotton', 'pulses', 'oilseeds', 'vegetables', 'fruits', 'tomato', 'potato', 'onion', 'chilli', 'turmeric', 'ginger']
            for crop in crops:
                if crop in query_lower:
                    st.session_state.bot.user_crop = crop.title()
                    break
            
            # Mark as initialized
            st.session_state.bot.is_initialized = True
            
            # Try processing again
            response = st.session_state.bot.process_query(query)
        
        return response
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def main():
    # Main container
    # st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header section
    st.markdown("""
    <div class="header-section">
        <h1>🌾 AGRIBOT</h1>
        <p style="font-size: 1.2rem; margin: 0;">Your smart farming partner for quicker, sharper, and greener decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info section
    st.markdown("""
    <div class="info-section">
        <h3>🤖 About This Bot</h3>
        <p>This AI-powered agricultural advisor helps farmers with:</p>
        <div class="feature-grid">
            <div class="feature-card">
                <h4>💰 Price Information</h4>
                <p>Get latest mandi prices for crops across India</p>
            </div>
            <div class="feature-card">
                <h4>🌤️ Weather Advice</h4>
                <p>Weather-based farming recommendations</p>
            </div>
            <div class="feature-card">
                <h4>📋 Government Policies</h4>
                <p>Information about agricultural schemes</p>
            </div>
            <div class="feature-card">
                <h4>🌾 Farming Tips</h4>
                <p>General agricultural advice and best practices</p>
            </div>
        </div>
        <p><strong>💡 Tip:</strong> You can ask questions in both English and Hindi. The bot will automatically detect your language and respond accordingly. Just mention your city or crop in your question!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Bot status and initialization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat with Agricultural Advisor")
    
    with col2:
        if st.session_state.bot:
            st.markdown('<span class="status-badge status-online">🟢 Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">🔴 Offline</span>', unsafe_allow_html=True)
            if st.button("Start Bot"):
                if initialize_bot():
                    st.success("Bot started successfully!")
                    st.rerun()
    
    # Chat input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Ask me anything about agriculture:",
        placeholder="Examples:\n• गेहूं का भाव क्या है?\n• What is the weather like in Mumbai?\n• PM Kisan scheme details\n• Rice prices in Kanpur\n• मौसम कैसा है?\n• Wheat prices in Delhi",
        height=120
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Send Message", type="primary"):
            if not st.session_state.bot:
                st.error("Please start the bot first!")
            elif user_input.strip():
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Get bot response with fallback handling
                with st.spinner("🤖 Processing your question..."):
                    try:
                        response = process_query_with_fallback(user_input)
                        st.session_state.chat_history.append({
                            "role": "bot",
                            "content": response,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.chat_history.append({
                            "role": "bot",
                            "content": error_msg,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                
                st.rerun()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col3:
        if st.button("Example Questions"):
            st.info("""
            **Try asking:**
            - गेहूं का भाव क्या है?
            - What is the weather like in Mumbai?
            - PM Kisan scheme details
            - Rice prices in Kanpur
            - मौसम कैसा है?
            - Best time to plant wheat
            - Wheat prices in Delhi
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("📝 Conversation History")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message user-message">
                    <strong>You ({message['timestamp']}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message bot-message">
                    <strong>Bot ({message['timestamp']}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 Start a conversation by asking a question above!")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat container
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <p><strong>🌾 Agricultural Advisor Bot</strong> | Powered by AI & Agricultural Data</p>
        <p style="font-size: 0.9rem; color: #666;">Supports English & Hindi | Real-time Price Data | Weather Insights | Policy Information</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main container

if __name__ == "__main__":
    main()